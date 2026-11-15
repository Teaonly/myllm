// ===========================================================================
// dq.cu — 量化块反量化原语（跨主题复用，独立成文件）
// ===========================================================================
// 全模型量化权重的解码原语：embedding 词表（embed.cu 的 gather）与量化
// GEMM 前置（gdn.cu 的权重 → f16）共用，故从 embedding 主题拆出、原型进
// all.cuh。
// 出处：Strata src/kernels/cuda/iq_kernels.cu——llama.cpp dequantize 的
// 64 线程角色折到 32 线程（tid 与 tid+32 各跑一半）。dq_q6_k 原样搬自
// :3196，dq_q4_k 搬自 :3098（同族同签名）。
// 出参是 Strata 的模板 dst_t：实例化 float（embedding，逐值原样）与
// __half（GEMM 前置，RNE 舍入——Strata dequant_bf16.cu put(H16*) 同款）。
// ===========================================================================
#include "all.cuh"

namespace kern {

namespace {

// 输出侧类型转换（Strata iq_kernels.cu:2929）
template <typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template <> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template <> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half_rn(v); }

// Q4_K 的 6bit 尺度/最小值解码（iq_kernels.cu:3089，Q4_K/Q5_K 共用）
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// (行, 超块) 一个 block：dq 原语进 shared 256 个 f16，再整块写出。f16 值与
// f32 路径同源（同一表达式），只多一步 RNE——投影 GEMM 的权重侧契约
__global__ void dq_rows_q6k_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                   uint16_t* __restrict__ out, int64_t k) {
    __shared__ uint16_t buf[256];
    const int64_t row = blockIdx.y;
    dq_q6_k<__half>(w + (size_t) row * row_bytes, blockIdx.x, (__half*) buf, threadIdx.x);
    __syncthreads();
    uint16_t* dst = out + row * k + (int64_t) blockIdx.x * 256;
    for (int i = threadIdx.x; i < 256; i += 32) dst[i] = buf[i];
}

__global__ void dq_rows_q4k_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                   uint16_t* __restrict__ out, int64_t k) {
    __shared__ uint16_t buf[256];
    const int64_t row = blockIdx.y;
    dq_q4_k<__half>(w + (size_t) row * row_bytes, blockIdx.x, (__half*) buf, threadIdx.x);
    __syncthreads();
    uint16_t* dst = out + row * k + (int64_t) blockIdx.x * 256;
    for (int i = threadIdx.x; i < 256; i += 32) dst[i] = buf[i];
}

}  // namespace

// scales 的 sc[0/2/4/6] 跨 2 来自 llama.cpp 布局里 q/k 两半的交错。
template <typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const Q6KBlock* x = (const Q6KBlock*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int64_t ip = tt / 32;                 // 0 or 1
        const int64_t il = tt - 32 * ip;            // 0...31
        const int64_t is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d = __half2float(x[ibs].d);
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0]   = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32]  = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64]  = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96]  = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}

// Q4_K（iq_kernels.cu:3098）：每线程管 8 值（y[l]/y[l+32]，l<4），值 =
// (d·sc)·nibble − dmin·m——带减项，与 Q6_K 的纯乘不同型
template <typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const Q4KBlock* x = (const Q4KBlock*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ibs].dm);
    const float dmin = __high2half(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}

// 显式实例化（定义只在本文件，跨 .cu 引用走设备链接 -dlink）
template __device__ void dq_q6_k<float>(const void*, int64_t, float*, int);
template __device__ void dq_q6_k<__half>(const void*, int64_t, __half*, int);
template __device__ void dq_q4_k<float>(const void*, int64_t, float*, int);
template __device__ void dq_q4_k<__half>(const void*, int64_t, __half*, int);

// ---- host 小入口：量化权重整块 → f16（投影 GEMM 的前置，Strata native_proj
//      的 dequant 步同位：dequant_f16 后紧接 f16 GEMM，gemm.cu:1456）----

void do_dq_q6k_f16(const uint8_t* w, int64_t rows, int64_t k, uint16_t* out, cudaStream_t stream) {
    vt_assert(k % 256 == 0, "do_dq_q6k_f16: K must be a multiple of 256 (Q6_K superblock)");
    dq_rows_q6k_kernel<<<dim3((unsigned) (k / 256), (unsigned) rows), 32, 0, stream>>>(
        w, (size_t) (k / 256) * 210, out, k);
    check_launch("dq_rows_q6k");
}

void do_dq_q4k_f16(const uint8_t* w, int64_t rows, int64_t k, uint16_t* out, cudaStream_t stream) {
    vt_assert(k % 256 == 0, "do_dq_q4k_f16: K must be a multiple of 256 (Q4_K superblock)");
    dq_rows_q4k_kernel<<<dim3((unsigned) (k / 256), (unsigned) rows), 32, 0, stream>>>(
        w, (size_t) (k / 256) * 144, out, k);
    check_launch("dq_rows_q4k");
}

}  // namespace kern
