// ===========================================================================
// gr.cu — 超连接（hyper-connection）读/写主题：gr_norm_rs / gr_silu /
//         gr_mix_r / gr_write 四个 kernel + bf16 GEMM（cuBLAS）+
//         do_gr_read / do_gr_write 发射器
// ===========================================================================
// 一个 half（attn 或 ffn）的残差读，计算逐字跟随 Strata：
//   kernel  src/prefill/kernels.cu  gr_norm_rs_kernel(:129) gr_silu_kernel(:269)
//           gr_mix_r_kernel(:148)，及设备件 bf/sigm/warp_sum/block_sum
//   GEMM    src/prefill/gemm.cu:1160 —— Gemm::bf16 在 sm_80+ 的默认路径：
//           cublasGemmEx(bf16×bf16→f32, COMPUTE_32F, GEMM_DEFAULT)，
//           handle 挂 32 MiB workspace + DEFAULT_MATH（Strata prefill.cpp:1276
//           GEMM_WS 同配置；workspace 参与 GEMM_DEFAULT 的算法挑选，跟随
//           Strata 保住可逐位对拍的同一调用形态）
// 序列（prefill.cpp:3727-3784 的编排，六步）：
//   1. gr_norm_rs ：逐流 RMSNorm → 行尺度 grs + bf16 像 xn16（不写 f32 xn，
//      gr_mix_r 按同序重算——Strata F-1，省一整块 T×D f32）
//   2. down GEMM  ：xn16 (T,D) × wd (LR,D)ᵀ → lo (T,LR) f32
//   3. gr_silu    ：lo16 = bf16(silu(lo / HC))——/4 在 silu 里面
//   4. up GEMM    ：lo16 (T,LR) × wu (D,LR)ᵀ → gated (T,D)
//   5. inject GEMM：xn16 (T,D) × wi (HC,D)ᵀ → inj (T,HC)——inject 是每 token
//      的真投影（10240→4），不是每层常量，block_inject_weight 这名字骗人
//   6. gr_mix_r   ：mixed = Σ_c (R·grs·w)_c · σ(gated_c) / HC——均值不是和
// 写回（一个 half 的收尾，prefill.cpp:5612 段的编排）：
//   gr_write：R[t,c,:] += bo[t,:]·2σ(inj[t,c]/HC)——块输出按流增益落残差栈
//   （kernels.cu:300）。未搬分叉：F-2 融合 gr_write_norm_rs（:174，写回 +
//   下一半的 gr_norm_rs 一遍过，same bits——层循环接入时若要，须给
//   do_gr_read 开"norm 已跑"模式）与 cvec steer 融合（:211，控制向量专属）
// 权重契约（IQ3_S 包 tensor-allocation 实测 + Strata ne0/ne1 断言反推）：
//   norm  (HC,N)  F32   键内即 1+w（hc_norm 存 1+w，乘即用，不另加 1）
//   down  (LR,D)  BF16  行主序按出维排：w[oy*K+k]，K=D
//   up    (D,LR)  BF16  同上，K=LR
//   inject(HC,D)  BF16  同上，K=D
//   尺寸哨兵在 prefill.cpp 的组装点（张量字节账对不上立刻叫）。
// gr_parity 钉死、一条都不能错的读法：
//   - RMS 逐流归（每流 2560，不是整栈 10240）；rs = rsqrtf(ss/N + eps)，
//     eps 加在均值上（GDN l2_norm 是加在已缩放平方范数上——两套约定，别串）
//   - xn16/lo16 的 bf16 像是契约（RNE 舍入 bf），GEMM 吃像、不吃 f32
//   - SiLU 在 lo、sigmoid 在 gate；σ 用 __expf
//   - 归约序：线程跨步累加 → warp_sum 洗牌树 → 块内 warp 部分和再洗牌树，
//     逐字照搬（换序即换位）
// 未搬的 Strata 分叉（都在 CUDA 默认路径之外，prefill.cpp:3724 的条件里）：
//   FP16 像（gfx11 rocBLAS 专属）、BF16X2 低像（对拍仪表）、WMMA 融合
//   （STRATA_PF_HCDOWN / gr_upmix）、XN_PAD 行距——本引擎一律 ldx=D。
// ===========================================================================
#include "all.cuh"
#include "qwen3.8-next.h"
#include "vt/vt.hpp"    // vt_panic

#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace kern {

// 几何短名（Strata kernels.cu:24 同款；kernel 体内直接用）
constexpr int N = model::HIDDEN;      // 2560
constexpr int HC = model::HC;         // 4
constexpr int D = model::HC_DIM;      // 10240 = HC*N
constexpr int LR = model::LOWRANK;    // 320

namespace {

// ---- 设备件：逐字取自 Strata src/prefill/kernels.cu ----

// f32 → bf16，RNE 舍入（kernels.cu:40 bf）
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// f32 → f16，RNE 舍入（kernels.cu:46 hf）——native 投影的 f16 像契约
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// 块宽 ≤1024 的块内和，结果广播（kernels.cu:95 block_sum）
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}

unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// cuBLAS handle：懒建、常驻（绑建当时的当前设备——与 execStream 同设备，
// scheduler 不会中途换卡）。32 MiB workspace + DEFAULT_MATH 同 Strata。
struct GrBlas {
    static constexpr size_t kWs = 32ull << 20;   // Strata prefill.cpp:1276 GEMM_WS
    cublasHandle_t h = nullptr;
    void* ws = nullptr;
    GrBlas() {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) vt_panic("kern: gr cublasCreate failed");
        if (cudaMalloc(&ws, kWs) != cudaSuccess) {
            cudaGetLastError();
            vt_panic("kern: gr cublas workspace cudaMalloc failed");
        }
        cublasSetWorkspace(h, ws, kWs);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    }
    ~GrBlas() {
        if (h) cublasDestroy(h);
        if (ws) cudaFree(ws);
    }
};

// Y (T,N) f32 行主序 = X (T,K) bf16 行主序 × W (N,K) bf16 行主序的转置。
// 列主视角 Y^T[N,T] = Wᵀ·X^T，与 Strata gemm.cu:1160 同字面（ldx=K、ldy=N）
void gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t Nn, int64_t K, cudaStream_t st) {
    static GrBlas blas;   // 首次调用即建，进程存活期内常驻
    const float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(blas.h, st);
    const cublasStatus_t s = cublasGemmEx(blas.h, CUBLAS_OP_T, CUBLAS_OP_N, (int) Nn, (int) T, (int) K,
                                          &alpha, W, CUDA_R_16BF, (int) K,
                                          X, CUDA_R_16BF, (int) K,
                                          &beta, Y, CUDA_R_32F, (int) Nn,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) vt_panic("kern: gr cublasGemmEx failed");
}

}  // namespace

// ---- kernels：Strata 原文的裁直版（丢 *_lo 仪表、f16img、ldx 哫距）----

// 逐流 RMSNorm → 行尺度 + bf16 像（kernels.cu:129，ldx=D、f16img=false 定版）
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * HC + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    const int64_t xo = (row / HC) * D + (int64_t) c * N;   // 像的行（token 步长 D）
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn16[xo + d] = bf(v);
    }
}

// lo16 = bf16(silu(lo / HC))（kernels.cu:269）
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    const float v = x / (1.0f + __expf(-x));
    lo16[i] = bf(v);
}

// mixed = Σ_c (R·rs·w)_c · σ(gate_c) / HC（kernels.cu:148；xn 按同序复算，
// 与 gr_norm_rs 的值逐位一致——F-1 的前提）。出三像：f32 本值 + bf16 像
// （router/α/β 契约）+ f16 像 mixed_h（native 投影契约，hf RNE）
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                uint16_t* __restrict__ mixed_h, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm 的值，bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    mixed16[i] = bf(s);
    mixed_h[i] = hf(s);
}

// R[t,c,:] += bo[t,:]·2σ(inj[t,c]/HC)（kernels.cu:300）。/HC 在 σ 里面
// （与 gr_silu 同约定：增益是 2·σ(注入/流数)）
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}

// ---- 发射器：一个 half 的完整读，六步按序 ----
void do_gr_read(volatile int* inputLeft, volatile int* inputRight,
                const float* wNorm,     // (HC,N)  f32   blk.%%.hc_{attn,ffn}_norm.weight
                const uint16_t* wDown,  // (LR,D)  bf16
                const uint16_t* wUp,    // (D,LR)  bf16
                const uint16_t* wInj,   // (HC,D)  bf16
                const float* R,         // (T,HC,N) f32 —— do_embed 的 res
                float* grs,             // (T,HC)  f32
                uint16_t* xn16,         // (T,D)   bf16
                float* lo, uint16_t* lo16,   // (T,LR)
                float* gated,           // (T,D)   f32
                float* inj,             // (T,HC)  f32
                float* mixed, uint16_t* mixed16, uint16_t* mixedh,  // (T,N) f32 + bf16/f16 两像
                cudaStream_t stream) {
    const int cs = model::CS;
    const int left = *inputLeft, right = *inputRight;
    const int T = right - left > cs ? cs : right - left;
    if (T <= 0) return;

    // 1. 逐流 norm → grs + xn16（grid = T*HC 行，块 256——Strata 发射同款）
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, stream>>>(R, wNorm, model::RMS_EPS, grs, xn16);
    check_launch("gr_norm_rs");

    // 2. down：xn16 (T,D) × wd (LR,D)ᵀ → lo (T,LR)
    gemm_bf16(xn16, wDown, lo, T, LR, D, stream);

    // 3. silu：lo → lo16（/HC 在里面）
    gr_silu_kernel<<<blocks_for((int64_t) T * LR), 256, 0, stream>>>(lo, lo16, (int64_t) T * LR);
    check_launch("gr_silu");

    // 4. up：lo16 (T,LR) × wu (D,LR)ᵀ → gated (T,D)
    gemm_bf16(lo16, wUp, gated, T, D, LR, stream);

    // 5. inject：xn16 (T,D) × wi (HC,D)ᵀ → inj (T,HC)
    gemm_bf16(xn16, wInj, inj, T, HC, D, stream);

    // 6. mix：均值出 mixed（f32 + bf16/f16 两像：bf16 喂 router/α/β，f16 喂 native 投影）
    gr_mix_r_kernel<<<blocks_for((int64_t) T * N), 256, 0, stream>>>(R, grs, wNorm, gated, mixed, mixed16, mixedh, T);
    check_launch("gr_mix_r");
}

// ---- 发射器：残差写回（一个 half 的收尾，一步）----
void do_gr_write(volatile int* inputLeft, volatile int* inputRight,
                 const float* bo,   // (T,N) 块输出：attn 半 = GDN 的 ssm_out 出（pf.emb 别名）
                 const float* inj,  // (T,HC) 本半的每流注入（do_gr_read 步 5 的出）
                 float* R,          // (T,HC,N) 残差栈，就地累加
                 cudaStream_t stream) {
    const int cs = model::CS;
    const int left = *inputLeft, right = *inputRight;
    const int T = right - left > cs ? cs : right - left;
    if (T <= 0) return;
    gr_write_kernel<<<blocks_for((int64_t) T * D), 256, 0, stream>>>(R, bo, inj, HC, T);
    check_launch("gr_write");
}

}  // namespace kern
