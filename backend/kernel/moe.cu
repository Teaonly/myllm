// ===========================================================================
// moe.cu — MoE 主题：路由 → 分组 → 路由专家 → 共享专家 → combine（层 0）
// ===========================================================================
// 一个 MoE 块的计算（Strata prefill.cpp:4107-5524 的默认路径，逐字跟随）：
//   路由    logits (T,NE) = mixed_bf × ffn_gate_inpᵀ     （prefill.cpp:4115
//           bf16_proj）→ route_kernel（kernels.cu:2218）：softmax + 逐 rank
//           贪心 top-k（同分取小 id），权重在选中集内归一 → ids/w (T,K)
//   分组    ids 回读 host，按专家计数/排布（prefill.cpp:4331-4543）：slot
//           （(t,k)→Dm 行号）与 src（Dm 行→token 号）表，上传
//   gather  xs (T·K,N) f16 = mixedh 按 src 收行（kernels.cu:2307）
//   专家    逐专家（prefill.cpp:5374-5402 默认 FP16 路径）：gate|up 反量化
//           成交错 f16（iq_kernels.cu:3343 dequant_gu_kernel）→ GEMM →
//           SwiGLU（kernels.cu:2282 swiglu_il_kernel）→ down 反量化
//           （:3329 dequant_flat_kernel）→ GEMM → Dm 行
//   共享    三个 native 投影 + swiglu_pair（kernels.cu:2300）+ 标量门
//           σ(mixed·gate_inp_shexp)（prefill.cpp:4178-4186）
//   combine bo = Σ_k w_k·Dm[slot_k] + shared·σ(sg)（kernels.cu:2315）
//           ——shared 与 bo 同缓冲原地写（每线程只读写自己的 i，安全）
// 层 0 类型实测（GGUF 头点名 + mats.cpp 字节哨兵）：路由/标量门 BF16；
// 专家 gate/up IQ3_XXS、down IQ4_NL；共享 gate Q4_K、up IQ4_XS、down
// IQ4_NL。层循环接入时按 48 层类型表参数化（见剧本 MoE 段注释：gate/up
// 四型 IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS，down 两型 IQ4_NL/Q2_0）。
//
// ---- 未搬的分叉（Strata 有、这里不走；接性能阶段再评估）----
//   - moe_combine4_kernel（kernels.cu:2334）：float4 向量化 combine，注释
//     明言与 moe_combine_kernel 逐位相同——默认走它，我们走基线版，bits 等价
//   - gemm_iq_f16_grouped（iq_kernels.cu:6667）：16 专家一组、超块进共享
//     内存一次解码的融合 WMMA GEMM（Strata 的性能路径，STRATA_PF_WMMA）；
//     我们用逐专家"反量化→cublas f16"替代——Strata 自家默认 FP16 路径同款
//   - MMQ（q8_1 激活）、blob 2bit（blob_dequant_kernel）、NVFP4 尾尺度、
//     APR Hadamard 折叠、流式 staging/专家缓存、CPU/peer 分担、peer 求和
//     combine（moe_combine_peer*）、scatter_add 路径——本包全驻留、单卡，
//     全部 N/A；STRATA_DBG_* 研究钩子不搬
//   - route 的 REG 模板按 NE 分派（kernels.cu:3108）：NE=512 → REG=16 定版
// ===========================================================================
#include "all.cuh"
#include "iq_tables.cuh"
#include "qwen3.8-next.h"

#include <cfloat>
#include <cstdio>
#include <cstring>
#include <vector>

namespace kern {

namespace {

// σ 与 f16 转换（gr.cu/gdn.cu 同款设备件；hf_sat 逐字取 kernels.cu:65——
// 饱和到 ±65504 防 SwiGLU 大激活变 inf 再进 down 投影成 NaN，NaN 原样透传）
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ uint16_t hf_sat(float f) {
    return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));
}
// warp 归约（gr.cu 同款；route_kernel 两处用）
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// ---------------------------------------------------------------------------
// dq 原语：一个超块 → yy 起 256 值，32 线程轮转（llama.cpp 公式，逐字取
// iq_kernels.cu:2968/:3018/:3052；出参在本文件恒 __half——所有读者都是
// f16 GEMM，f32 特化无场景，模板省去）
// ---------------------------------------------------------------------------

// IQ3_XXS：d + qs[96]（后 32 字节是 8 个子块的 uint16 尺度对）
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, __half* yy, int tid) {
    const IQ3XXSBlock* x = (const IQ3XXSBlock*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    __half* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + 64) + 2 * ib;   // QK_K/4
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = __float2half_rn(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = __float2half_rn(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}

// IQ4_NL：256 值 = 8 个 32 值小块，d + qs[16] nibble 码 → 非线性码本
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, __half* yy, int tid) {
    const IQ4NLBlock* x = (const IQ4NLBlock*) vx + ibs * 8;   // QK_K/QK4_NL
    const int64_t il = tid / 8, ib = tid % 8;
    __half* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = __float2half_rn(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = __float2half_rn(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}

// IQ4_XS 的 dq 原语已挪进 dq.cu（共用：嵌入侧要 f32，见 all.cuh 声明）；
// moe.cu 的 dq_iq4_nl / 交错反量化照旧用 __half 版实例

// 平坦反量化：超块 i → y + 256i（iq_kernels.cu:3329 定型版，type 具体化）
__global__ void dq_flat_iq4nl_kernel(const uint8_t* __restrict__ vx, __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_iq4_nl(vx, i, y + i * 256, threadIdx.x);
}
__global__ void dq_flat_iq4xs_kernel(const uint8_t* __restrict__ vx, __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_iq4_xs(vx, i, y + i * 256, threadIdx.x);
}

// gate|up 交错反量化（iq_kernels.cu:3343）：第 i 超块 → 角色 parity 的交错
// 行 2r+parity；grid (n_ff·per_row, 2)，32 线程
__global__ void dq_gu_iq3xxs_kernel(const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up,
                                    int64_t per_row, __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_iq3_xxs(parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * 256, threadIdx.x);
}

// ---------------------------------------------------------------------------
// MoE kernels（逐字取 Strata src/prefill/kernels.cu；N 在那边是编译期常量，
// 这里按 myllm 惯例走形状参数）
// ---------------------------------------------------------------------------

// softmax + 逐 rank 贪心 top-10，权重在选中集内归一（kernels.cu:2218）。
// 一个 warp 一个 token：REG×32 = NE 个 logits；同分取小 id（:2247 的 tie
// 规则保位稳定）；isnan 防护与 selected_sum 下限 2^-14 逐字保留
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}

// SwiGLU 交错版（kernels.cu:2282）：gu 行内 gate|up 交错（g=gu[2k], u=gu[2k+1]），
// h16 = f16(silu(g)·u)，饱和
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}

// SwiGLU 分离版（kernels.cu:2300）：共享专家用（gate/up 两投影分离缓冲）
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}

// 按 src 收行：一个 uint4（8 个 f16）一线程（kernels.cu:2307）
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}

// combine（kernels.cu:2315）：bo = Σ_k w_k·Dm[slot_k] + shared·σ(sg)。
// row_sd（NVFP4 的行尺度）恒 nullptr——本包无此形
__global__ void moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                   const float* __restrict__ w, const float* __restrict__ shared,
                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    const int64_t t = i / n, d = i % n;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const int64_t r = slot[t * 10 + k];
        const float v = Dm[r * n + d];
        s = fmaf(w[t * 10 + k], v, s);
    }
    bo[i] = s + shared[i] * sigm(sg[t]);
}

}  // namespace

// ---------------------------------------------------------------------------
// 发射器：一个 MoE 块（层 0）的完整序列
// ---------------------------------------------------------------------------
void do_moe_layer(volatile int* inputLeft, volatile int* inputRight,
                  const uint16_t* wRouter,   // (NE,N)  BF16  blk.0.ffn_gate_inp.weight
                  const uint16_t* wShGate,   // (N,)    BF16  blk.0.ffn_gate_inp_shexp.weight
                  const uint8_t* expGate,    // (NE,FF,N) IQ3_XXS 块  blk.0.ffn_gate_exps.weight
                  const uint8_t* expUp,      // (NE,FF,N) IQ3_XXS 块  blk.0.ffn_up_exps.weight
                  const uint8_t* expDown,    // (NE,N,FF) IQ4_NL  块  blk.0.ffn_down_exps.weight
                  const uint8_t* shGate,     // (FF,N)  Q4_K   blk.0.ffn_gate_shexp.weight
                  const uint8_t* shUp,       // (FF,N)  IQ4_XS blk.0.ffn_up_shexp.weight
                  const uint8_t* shDown,     // (N,FF)  IQ4_NL blk.0.ffn_down_shexp.weight
                  const uint16_t* mixed16,   // (T,N) bf16 像——路由/标量门的 GEMM 入参
                  const uint16_t* mixedh,    // (T,N) f16 像——专家/共享投影的 GEMM 入参
                  float* logits,             // (T,NE) 路由 logits
                  int32_t* ids,              // (T,K)  专家 id
                  float* w,                  // (T,K)  归一权重
                  int32_t* slot,             // (T·K)  (t,k)→Dm 行
                  int32_t* src,              // (T·K)  Dm 行→token
                  uint16_t* xs,              // (T·K,N) f16 收行
                  float* gu,                 // (T·K,2FF) f32 gate|up 行
                  uint16_t* hh,              // (T·K,FF) f16 SwiGLU 中间
                  float* dm,                 // (T·K,N) f32 各专家输出（"pf.moe.parts"）
                  uint16_t* dqGu,            // (2FF,N) f16 专家反量化 scratch（gate|up 交错）
                  uint16_t* dqD,             // (N,FF)  f16 down 反量化 scratch
                  float* sgate,              // (T,FF)  f32 共享 gate 投影
                  float* sup,                // (T,FF)  f32 共享 up 投影
                  uint16_t* shh,             // (T,FF)  f16 共享 SwiGLU 中间
                  float* shared,             // (T,N)   f32 共享专家输出 = bo（原地 combine）
                  float* sg,                 // (T,)    f32 标量门
                  float* bo,                 // (T,N)   f32 块输出（与 shared 同缓冲）
                  cudaStream_t stream) {
    const int cs = model::CS;
    const int left = *inputLeft, right = *inputRight;
    const int T = right - left > cs ? cs : right - left;
    if (T <= 0) return;
    const int64_t N = model::HIDDEN, FF = model::MOE_INT, K = model::TOP_K, NE = model::N_EXPERTS;

    // 1. 路由 GEMM + top-k（prefill.cpp:4115-4116；REG=16 即 NE=512）
    gemm_bf16(mixed16, wRouter, logits, T, NE, N, NE, stream);
    route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, stream>>>(logits, ids, w, T);
    check_launch("route");

    // 2. host 分组（prefill.cpp:4331-4543 的单卡全驻留形态）：ids 回读 →
    //    逐专家计数 → 行偏移前缀 → slot/src 表 → 上传。表内存量 T·K 项，
    //    逐拍现建（层循环接入后若成热点再换驻留 scratch）
    std::vector<int32_t> ids_h((size_t) T * K), slot_h((size_t) T * K), src_h((size_t) T * K);
    std::vector<int32_t> cnt(NE), off(NE + 1);
    if (cudaMemcpyAsync(ids_h.data(), ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        vt_panic("kern: moe routed ids download failed");
    std::fill(cnt.begin(), cnt.end(), 0);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t e = ids_h[(size_t) i];
        if (e < 0 || e >= NE) vt_panic("kern: moe routed id out of range");
        ++cnt[(size_t) e];
    }
    {
        int32_t r = 0;
        for (int32_t e = 0; e < NE; ++e) { off[(size_t) e] = r; r += cnt[(size_t) e]; }
        off[(size_t) NE] = r;
    }
    {
        std::vector<int32_t> fill(off.begin(), off.end() - 1);
        for (int64_t i = 0; i < T * K; ++i) {
            const int32_t e = ids_h[(size_t) i];
            const int32_t p = fill[(size_t) e]++;
            slot_h[(size_t) i] = p;
            src_h[(size_t) p] = (int32_t) (i / K);
        }
    }
    if (cudaMemcpyAsync(slot, slot_h.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(src, src_h.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        vt_panic("kern: moe grouping tables upload failed");

    // 3. 收行：xs (T·K,N) f16 ← mixedh（prefill.cpp:4784）
    gather_rows16_kernel<<<blocks_for((int64_t) T * K * (N / 8)), 256, 0, stream>>>(mixedh, src, xs, T * K, N);
    check_launch("gather_rows16");

    // 4. 共享专家（prefill.cpp:4178-4186）：三投影 + SwiGLU + 标量门。
    //    dqGu 前半 (FF,N) 给 gate、dqD 给 up（尺寸恰好 FF·N）；down 复用 dqD
    do_dq_q4k_f16(shGate, FF, N, dqGu, stream);
    gemm_f16(mixedh, dqGu, sgate, T, FF, N, FF, stream);
    do_dq_iq4xs_flat_f16(shUp, (int64_t) FF * N, dqD, stream);
    gemm_f16(mixedh, dqD, sup, T, FF, N, FF, stream);
    swiglu_pair_kernel<<<blocks_for((int64_t) T * FF), 256, 0, stream>>>(sgate, sup, shh, T);
    check_launch("swiglu_pair");
    do_dq_iq4nl_flat_f16(shDown, (int64_t) N * FF, dqD, stream);
    gemm_f16(shh, dqD, shared, T, N, FF, N, stream);
    gemm_bf16(mixed16, wShGate, sg, T, 1, N, 1, stream);

    // 5. 路由专家逐个（prefill.cpp:5374-5402 默认 FP16 路径）：每个专家
    //    反量化 → GEMM → SwiGLU → GEMM，行落在分组给的 [off[e], off[e]+cnt[e])
    const int64_t gu_bytes = (int64_t) FF * (N / 256) * 98;   // IQ3_XXS 每专家 gate/up
    const int64_t d_bytes = (int64_t) N * (FF / 32) * 18;     // IQ4_NL 每专家 down
    for (int32_t e = 0; e < NE; ++e) {
        const int64_t ne = cnt[(size_t) e];
        if (ne == 0) continue;
        const int64_t o0 = off[(size_t) e];
        dq_gu_iq3xxs_kernel<<<dim3((unsigned) ((int64_t) FF * N / 256), 2), 32, 0, stream>>>(
            expGate + (int64_t) e * gu_bytes, expUp + (int64_t) e * gu_bytes, N / 256, (__half*) dqGu);
        check_launch("dq_gu_iq3xxs");
        gemm_f16(xs + o0 * N, dqGu, gu + o0 * 2 * FF, ne, 2 * FF, N, 2 * FF, stream);
        swiglu_il_kernel<<<blocks_for(ne * FF), 256, 0, stream>>>(gu + o0 * 2 * FF, hh + o0 * FF, ne);
        check_launch("swiglu_il");
        dq_flat_iq4nl_kernel<<<(unsigned) (N * FF / 256), 32, 0, stream>>>(expDown + (int64_t) e * d_bytes, (__half*) dqD);
        check_launch("dq_flat_iq4nl");
        gemm_f16(hh + o0 * FF, dqD, dm + o0 * N, ne, N, FF, N, stream);
    }

    // 6. combine（prefill.cpp:5524）：bo = Σ w·Dm[slot] + shared·σ(sg)，原地
    moe_combine_kernel<<<blocks_for((int64_t) T * N), 256, 0, stream>>>(dm, slot, w, shared, sg, bo, T, N);
    check_launch("moe_combine");
}

// ---------------------------------------------------------------------------
// 独立反量化小入口（共享专家/将来层循环用；实现在上，走具体类型不走分派）
// ---------------------------------------------------------------------------
void do_dq_iq4nl_flat_f16(const uint8_t* w, int64_t n, uint16_t* out, cudaStream_t stream) {
    if (n % 256 != 0) vt_panic("kern: iq4_nl flat dq needs n%256==0");
    dq_flat_iq4nl_kernel<<<(unsigned) (n / 256), 32, 0, stream>>>(w, (__half*) out);
    check_launch("dq_flat_iq4nl");
}
void do_dq_iq4xs_flat_f16(const uint8_t* w, int64_t n, uint16_t* out, cudaStream_t stream) {
    if (n % 256 != 0) vt_panic("kern: iq4_xs flat dq needs n%256==0");
    dq_flat_iq4xs_kernel<<<(unsigned) (n / 256), 32, 0, stream>>>(w, (__half*) out);
    check_launch("dq_flat_iq4xs");
}

}  // namespace kern
