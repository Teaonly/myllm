// ===========================================================================
// prefill.cpp — prefill kernel 的 host 侧发射器（nvcc -x cu 编译）
// ===========================================================================
// 两条 prefill 路径当前挂 dummy_kernel：真计算接入前的占位，模拟一段耗时
// 计算，末尾把完工值写进 outputCmd——host 轮询即知完工，值可分诊路径。
// ===========================================================================
#include "index.h"

#include <cstdio>

namespace kern {

namespace {

// 测试负载时长：每线程 FMA 依赖链步数。链是延迟界，SM 多寡几乎不影响
// 时长（每线程一条串行链，全线程并行推进）——要多久改这里。
// MX450 -O0 实测 20M ≈ 0.4s
const int kLoadIters = 20000000;

// dummy：每线程一条 FMA 依赖链拖时长，块首线程链完写旗。
// 语义是"最先完工的 block 落旗"（各块工作量一致，时差可忽略；exec 流
// 后续工作仍按流序等全体 block）——真计算的完工协议接入时再收紧
__global__ void dummy_kernel(int iters, volatile int* outputCmd, int outValue) {
    float a = 0.5f;
    for (int i = 0; i < iters; ++i) {
        a = fmaf(a, 1.0000001f, 1e-7f);
    }
    if (threadIdx.x == 0) {
        // a 末值量级 <10，乘 1e-30 取整恒为 0：恰好写 outValue，又消耗
        // 掉 a——不消耗的话循环会被优化掉，负载瞬间完成
        *outputCmd = outValue + (int)(a * 1e-30f);
    }
}

// 公共发射：全 SM 铺满（每 SM 8 block × 256 线程）。流与旗都从 staff
// 来（execStream 在 scheduler 侧已 wait 过 h2dEvent）。
// 完工旗契约（归调用方/scheduler）：发射前 host 把 outputCmd 清 0——
// 0 只由 host 写、完工值只由 kernel 写；重新发射前上批必须已完工
void launch_dummy(myllm::SchedulerStaff& staff, int outValue) {
    static int n_sm = 0;    // 一次性查 SM 数
    if (n_sm == 0) {
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
    }
    dummy_kernel<<<n_sm * 8, 256, 0, staff.execStream>>>(kLoadIters,
                                                         staff.outputCmd, outValue);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kern: dummy launch failed: %s\n",
                     cudaGetErrorString(err));
        vt_panic("kern: dummy launch failed");
    }
}

}  // namespace

void do_prefill_chunking(CudaEnv* env, myllm::SchedulerStaff& staff) {
    (void)env;    // 真实版：env->hash() 取形状；输入窗口 [inputLeft,inputRight) 定工作量
    launch_dummy(staff, 1);    // 完工值 1 = chunking 一块喂完
}

void do_prefill_chunked(CudaEnv* env, myllm::SchedulerStaff& staff) {
    (void)env;
    launch_dummy(staff, 2);    // 完工值 2 = 一拍吃完路径完工
}

}  // namespace kern
