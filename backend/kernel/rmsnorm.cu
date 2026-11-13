// ===========================================================================
// rmsnorm.cu — 分组 RMSNorm（接口契约见 cuda_kernels.cuh）
// ===========================================================================
//
// 规格来源（三处钉死，写 kernel 勿凭记忆）：
//   公式   transformers modeling:Qwen4ExpTextRMSNorm.forward —— f32 域
//          x*rsqrt(mean(x²)+eps) 再乘 (1+w)；Llama 是 x*w、Qwen4Exp 是
//          (x*(1+w))，权重 1-centered（golden 里 w=0 即乘 1）；
//   分组   hc_norm/norm_* 构造带 group_size=hidden(2560)：10240 宽按 4 段
//          各自归一（残差流条数独立），非全宽一组——golden layer5 实测分组
//          diff ~bf16 舍入级、不分组 diff 1.2e0，混用即错模型；
//   数值   golden 计算全程 f32、落盘才压 bf16（gen_golden stow 是 detached
//          副本），所以 kernel f32 进出即可，对拍判 bf16 舍入量级。
//
// 实现选型：一 block 一段（row×组号打平成段号）。段长 group ≤ 10240
// (模型最大 2560)，256 线程跨两轮访存（sumsq 一轮、回写一轮）——段数据
// 10KB 级驻 L1/L2，二读代价可忽略，换来实现无寄存器上限。块内归约走
// warp shuffle + 共享内存两级（kThreads/32 ≤ 32，第二级一个 warp 串完）。
// ===========================================================================
#include "cuda_kernels.cuh"

#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

// 块内求和：warp shuffle 归约 → 各 warp 首线程落共享 → 0 号 warp 串归约
// → 结果写 warp[0]，全块同步后各自读。线程不足 kThreads 参与数的段
// （group < 256，如 indexer 的 128）多余线程贡献 0，不影响和。
__device__ __forceinline__ float block_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    __shared__ float warp[kThreads / 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) warp[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lane < kThreads / 32 ? warp[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) warp[0] = v;
    }
    __syncthreads();
    return warp[0];
}

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ y,
                               __nv_bfloat16* __restrict__ y16,
                               int width, int group, float eps) {
    const int per_row = width / group;             // 每行段数
    const size_t base = (size_t)blockIdx.x / per_row * width +  // 行首
                        (size_t)blockIdx.x % per_row * group;   // 段首
    const float* xs = x + base;

    float ss = 0.f;                                // 段内平方和（f32，契约）
    for (int i = threadIdx.x; i < group; i += kThreads) {
        const float v = xs[i];
        ss += v * v;
    }
    const float rms = rsqrtf(block_sum(ss) / group + eps);

    float* ys = y + base;
    for (int i = threadIdx.x; i < group; i += kThreads) {
        // w 逐通道按全局列号（段首+段内）取；1-centered：乘 (1+w)
        const float v = xs[i] * rms * (w ? 1.f + w[base % width + i] : 1.f);
        ys[i] = v;
        if (y16) y16[base + i] = __float2bfloat16(v);
    }
}

}  // namespace

namespace kern {

void rmsnorm(const float* x, const float* w, float* y, uint16_t* y16,
             int rows, int width, int group, float eps, void* stream) {
    if (rows <= 0 || width <= 0 || group <= 0 || width % group != 0 ||
        (group > 10240)) {
        std::fprintf(stderr, "kern::rmsnorm: bad shape rows=%d width=%d group=%d\n",
                     rows, width, group);
        std::abort();
    }
    const int segs = rows * (width / group);
    rmsnorm_kernel<<<segs, kThreads, 0, (cudaStream_t)stream>>>(
        x, w, y, reinterpret_cast<__nv_bfloat16*>(y16), width, group, eps);
}

}  // namespace kern
