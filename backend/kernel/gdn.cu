// ===========================================================================
// gdn.cu — GDN 线性注意力层主题（layer 0）：四投影 + gates + conv + 递归 +
//          输出投影 + do_gdn_layer 发射器
// ===========================================================================
// 一个 GDN 层 attn 半的完整计算，逐字跟随 Strata（CUDA 默认路径）：
//   序列     src/prefill/prefill.cpp:3803-3834 —— qkv → gate → α → β →
//            gates → conv → recurrence → out 投影，八步按序
//   kernel   src/prefill/kernels.cu  gdn_gates_kernel(:315) gdn_conv_kernel(:326)
//            gdn_conv_tiled_kernel(:344) gdn_conv_hist_kernel(:454)
//            gdn_l2_kernel(:472) gdn_rec_cols_kernel(:595) gdn_out_norm_kernel(:2200)
//            及设备件 sigm/warp_sum（hf/bf 同 gr.cu 一字不差）
//   GEMM     native 投影（qkv/gate/out）：权重 Q6_K/Q4_K → f16 反量化（dq.cu
//            原语）+ f16 GEMM——gemm.cu Gemm::native(:1346) 默认两步
//            dequant_f16 + f16(:1164)，cublasGemmEx(f16×f16→f32,
//            COMPUTE_32F, GEMM_DEFAULT)，handle 32 MiB workspace +
//            DEFAULT_MATH（Strata prefill.cpp:1276 GEMM_WS 同配置）；
//            α/β 是 bf16 GEMM（gemm.cu:1160，同 gr.cu）多一个 ldy——
//            (T,2HV) 的 α|β 合排交错写
// 权重契约（gsq 包 tensor-allocation 实测，layer 0；哨兵在 prefill.cpp）：
//   attn_qkv.weight   (C,N)   Q6_K   2100 B/行
//   attn_gate.weight  (ZV,N)  Q4_K   1440 B/行
//   ssm_out.weight    (N,ZV)  Q6_K   5040 B/行（K=ZV）
//   ssm_alpha/beta.weight (VH,N) BF16（行主序按出维排，K=N）
//   ssm_conv1d.weight (C,4)   F32    w[c*4+k]
//   ssm_norm.weight   (S,)    F32    输出 norm 的 γ（逐头共享）
//   ssm_dt.bias       (VH,)   F32    gate 的 softplus 偏置
//   ssm_a             (VH,)   F32    gate 的逐头尺度 A
// 状态契约（session 'gdn.S'/'gdn.conv' 的 layer 0 基址，就地推进）：
//   S 布局 (S,HV,S) = (k 行, v 头, v 列)，j 最快——kernels.cu:595 原文如此；
//   conv 历史 (C,3)，channel 主序 hist[c*3+i] = pre-conv 输入的最近 3 行
//   （当前行在 qkv 激活里，不占状态）
// 未搬的 Strata 分叉（都在 CUDA 默认路径之外）：
//   KVA state-only（STRATA_KVA_*）、HIP 融合 conv+recurrence、分块递归
//   （gdn_chunk.cu，Volta 实验构建）、keyhead/pipe 同位加速变体
//   （kernels.cu:801/638，"same bits"，量出来需要再搬）、conv 的 L2 融合
//   三款（STRATA_GDN_CONVL2）、W16 缓存（STRATA_PF_W16_CACHE_MIB——每拍
//   反量化同 Strata 默认，缓存是显存换带宽的可选项）
// ===========================================================================
#include "all.cuh"
#include "qwen3.8-next.h"
#include "vt/vt.hpp"    // vt_panic

#include <cublas_v2.h>

#include <cstdio>

namespace kern {

// 几何短名（Strata kernels.cu:25 同款；kernel 体内直接用）
constexpr int S = model::K_DIM;           // 128：状态维 / 头宽
constexpr int HK = model::K_HEADS;        // 16：k 头
constexpr int HV = model::V_HEADS;        // 48：v 头
constexpr int C = model::GDN_CONV_WIDTH;  // 10240 = q2048|k2048|v6144
constexpr int ZV = model::VALUE_DIM;      // 6144 = HV*S（z/y 宽）
constexpr float EPS = 1.0e-6f;            // Strata prefill.cpp:171（conv L2 与输出 norm 共用）

namespace {

// ---- 设备件：逐字取自 Strata src/prefill/kernels.cu ----

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// f32 → f16，RNE 舍入（kernels.cu:46 hf）——y16 契约
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }

unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// cuBLAS handle：懒建、常驻（契约同 gr.cu 的 GrBlas——32 MiB workspace +
// DEFAULT_MATH，Strata GEMM_WS 同配置；每主题一份，同形调用同一算法）
struct GdnBlas {
    static constexpr size_t kWs = 32ull << 20;   // Strata prefill.cpp:1276 GEMM_WS
    cublasHandle_t h = nullptr;
    void* ws = nullptr;
    GdnBlas() {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) vt_panic("kern: gdn cublasCreate failed");
        if (cudaMalloc(&ws, kWs) != cudaSuccess) {
            cudaGetLastError();
            vt_panic("kern: gdn cublas workspace cudaMalloc failed");
        }
        cublasSetWorkspace(h, ws, kWs);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    }
    ~GdnBlas() {
        if (h) cublasDestroy(h);
        if (ws) cudaFree(ws);
    }
};

// Y (T,Nn) f32 行主序 = X (T,K) f16 行主序 × W (Nn,K) f16 行主序的转置，
// 行距 ldy（gemm.cu:1164 默认形态：OP_T OP_N, f16×f16→f32, COMPUTE_32F,
// GEMM_DEFAULT；ldx=K）
void gemm_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t Nn, int64_t K, int64_t ldy,
              cudaStream_t st) {
    static GdnBlas blas;
    const float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(blas.h, st);
    const cublasStatus_t s = cublasGemmEx(blas.h, CUBLAS_OP_T, CUBLAS_OP_N, (int) Nn, (int) T, (int) K,
                                          &alpha, W, CUDA_R_16F, (int) K,
                                          X, CUDA_R_16F, (int) K,
                                          &beta, Y, CUDA_R_32F, (int) ldy,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) vt_panic("kern: gdn cublasGemmEx f16 failed");
}

// bf16 版（gemm.cu:1160，同 gr.cu 的 gemm_bf16 + ldy——α/β 交错写 ab 用）
void gemm_bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t Nn, int64_t K, int64_t ldy,
               cudaStream_t st) {
    static GdnBlas blas;
    const float alpha = 1.0f, beta = 0.0f;
    cublasSetStream(blas.h, st);
    const cublasStatus_t s = cublasGemmEx(blas.h, CUBLAS_OP_T, CUBLAS_OP_N, (int) Nn, (int) T, (int) K,
                                          &alpha, W, CUDA_R_16BF, (int) K,
                                          X, CUDA_R_16BF, (int) K,
                                          &beta, Y, CUDA_R_32F, (int) ldy,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s != CUBLAS_STATUS_SUCCESS) vt_panic("kern: gdn cublasGemmEx bf16 failed");
}

}  // namespace

// ---- kernels：Strata 原文的裁直版（丢 *_lo 仪表与未启用的分叉）----

// gate = softplus(αᵣ+dt)·A、β = σ(βᵣ)（kernels.cu:315）。softplus 上支
// v>20 直通（exp 溢出保护），log1pf 与 __expf 同源
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}

// 4 抽头因果卷积 + SiLU，串行版（kernels.cu:326）：一线程一通道走完整块，
// 历史随行走、块尾落回 hist
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}

// 分块版（kernels.cu:344）：卷积读输入不读自己的输出，64 token 一片各自
// 取前身（片内或 hist），片间独立——每元素同表达式同位；历史另由
// gdn_conv_hist_kernel 落账
constexpr int CONV_TILE = 64;
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}

// 块尾历史 = 最近 3 个 pre-conv 输入（块不够 3 时接旧历史，kernels.cu:454）
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}

// q/k 头逐头 L2（kernels.cu:472）：32 个 q/k 头，warp 和 + 4 部分和定序；
// eps 加在未缩放平方和上（与 RMS 的均值约定不同，别串）
__global__ void gdn_l2_kernel(float* __restrict__ h, float eps) {
    const int64_t t = blockIdx.y;
    const int head = blockIdx.x;
    float* x = h + t * C + head * S;
    const float v = x[threadIdx.x];
    float sq = warp_sum(v * v);
    __shared__ float part[4];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    x[threadIdx.x] = v * rsqrtf(ss + eps);
}

// 门控增量网络递归，列分块基线版（kernels.cu:595）。每块管一个 (v 头,
// 32 列片)；状态在寄存器，token 串行：kv = kᵀW → δ = β(v−g·kv) →
// W ← g·W + k·δᵀ → o = qᵀW，四行组部分和按 red[0..3] 定序相加（位序
// 契约）；oc 出预 norm 值 × rsqrt(S)
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;  // 列分块：值列/块、块数/头
__global__ void __launch_bounds__(CB * RG) gdn_rec_cols_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                              const float* __restrict__ gate,
                                                              const float* __restrict__ beta,
                                                              float* __restrict__ oc_out, int64_t T) {
    __shared__ float sk[S], sq[S], red[RG][CB];
    const int head = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * C;
        __syncthreads();
        if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[t * HV + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        __syncthreads();
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][c] = o;
        __syncthreads();
        if (rg == 0) oc_out[t * HV * S + head * S + col] = (red[0][c] + red[1][c] + red[2][c] + red[3][c]) * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}

// 输出门控 RMS norm（kernels.cu:2200）：y = oc·rsqrt(ss/S + eps)·γ·σ(z)，
// 只落 f16 像 y16（f32 归一值无读者，不写）
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                         int64_t ld16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[(size_t) t * ld16 + (size_t) head * S + col] = hf(v);
}

// ---- 发射器：一个 GDN 层（layer 0）的完整八步 ----
void do_gdn_layer(volatile int* inputLeft, volatile int* inputRight,
                  const uint8_t* wQkv,     // (C,N)   Q6_K 原始块  blk.0.attn_qkv.weight
                  const uint8_t* wGate,    // (ZV,N)  Q4_K          blk.0.attn_gate.weight
                  const uint8_t* wOut,     // (N,ZV)  Q6_K          blk.0.ssm_out.weight
                  const uint16_t* wAlpha,  // (VH,N)  BF16          blk.0.ssm_alpha.weight
                  const uint16_t* wBeta,   // (VH,N)  BF16          blk.0.ssm_beta.weight
                  const float* wConv1d,    // (C,4)   F32           blk.0.ssm_conv1d.weight
                  const float* wNorm,      // (S,)    F32           blk.0.ssm_norm.weight
                  const float* wDtBias,    // (VH,)   F32           blk.0.ssm_dt.bias
                  const float* wSsmA,      // (VH,)   F32           blk.0.ssm_a
                  const uint16_t* mixedh,  // (T,N)   f16 像——qkv/gate/out 的 GEMM 入参
                  const uint16_t* mixed16, // (T,N)   bf16 像——α/β 的 GEMM 入参
                  uint16_t* w16,           // 反量化 scratch（≥ C×N 个 f16）
                  float* qkv,              // (T,C)   f32 投影输出 = conv 输入
                  float* z,                // (T,ZV)  f32 输出门
                  float* ab,               // (T,2VH) f32 α|β 合排
                  float* gate,             // (T,VH)  f32 递归门
                  float* beta,             // (T,VH)  f32 递归 β
                  float* h,                // (T,C)   f32 conv+SiLU+L2 后的 q|k|v
                  float* y,                // (T,ZV)  f32 递归输出（norm 前）
                  uint16_t* y16,           // (T,ZV)  f16 norm 后像
                  float* bo,               // (T,N)   f32 层输出（ssm_out 投影）
                  float* state,            // 'gdn.S' layer 0 基址：(S,HV,S) f32
                  float* convHist,         // 'gdn.conv' layer 0 基址：(C,3) f32
                  cudaStream_t stream) {
    const int cs = model::CS;
    const int left = *inputLeft, right = *inputRight;
    const int T = right - left > cs ? cs : right - left;
    if (T <= 0) return;
    const int N = model::HIDDEN;

    // 1. qkv 投影：Q6_K → f16 → GEMM（mixedh (T,N) × wqkvᵀ → qkv (T,C)）
    do_dq_q6k_f16(wQkv, C, N, w16, stream);
    gemm_f16(mixedh, w16, qkv, T, C, N, C, stream);

    // 2. gate 投影：Q4_K → f16 → GEMM（→ z (T,ZV)）。层 0 实测 Q4_K
    //    （allocation 表）；层循环接入时按表参数化反量化类型
    do_dq_q4k_f16(wGate, ZV, N, w16, stream);
    gemm_f16(mixedh, w16, z, T, ZV, N, ZV, stream);

    // 3/4. α/β 投影：bf16 GEMM 交错写 ab 的 [0,VH) 与 [VH,2VH)（ldy=2VH）
    gemm_bf16(mixed16, wAlpha, ab, T, HV, N, 2 * HV, stream);
    gemm_bf16(mixed16, wBeta, ab + HV, T, HV, N, 2 * HV, stream);

    // 5. gates：gate = softplus(α+dt)·A、β = σ(βᵣ)
    gdn_gates_kernel<<<blocks_for((int64_t) T * HV), 256, 0, stream>>>(ab, wDtBias, wSsmA, gate, beta, T);
    check_launch("gdn_gates");

    // 6. conv + SiLU + q/k L2：T≤64 串行版（历史随行走）；否则分块版 +
    //    历史落账（Strata 默认分派同款）。L2 恒跑（32 个 q/k 头）
    if (T <= CONV_TILE) {
        gdn_conv_kernel<<<C / 128, 128, 0, stream>>>(convHist, qkv, wConv1d, h, T);
        check_launch("gdn_conv");
    } else {
        gdn_conv_tiled_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0, stream>>>(
            convHist, qkv, wConv1d, h, T);
        check_launch("gdn_conv_tiled");
        gdn_conv_hist_kernel<<<C / 128, 128, 0, stream>>>(convHist, qkv, T);
        check_launch("gdn_conv_hist");
    }
    gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S, 0, stream>>>(h, EPS);
    check_launch("gdn_l2");

    // 7. 递归（state 就地推进）+ 输出门控 norm（y → y16）
    gdn_rec_cols_kernel<<<HV * NCB, dim3(CB, RG), 0, stream>>>(state, h, gate, beta, y, T);
    check_launch("gdn_rec_cols");
    gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, stream>>>(z, wNorm, EPS, y, y16, ZV);
    check_launch("gdn_out_norm");

    // 8. out 投影：Q6_K（K=ZV）→ f16 → GEMM（y16 (T,ZV) × woutᵀ → bo (T,N)）
    do_dq_q6k_f16(wOut, N, ZV, w16, stream);
    gemm_f16(y16, w16, bo, T, N, ZV, N, stream);
}

}  // namespace kern
