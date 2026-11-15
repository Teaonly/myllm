// ===========================================================================
// dummy.cu — 占位负载主题：dummy_kernel 实现 + do_dummy host 发射器
// ===========================================================================
// 真计算接入前模拟一段耗时计算（每线程一条 FMA 依赖链拖时长）。排空探测
// 在 scheduler 侧用 cudaStreamQuery 查 exec 流（全 kernel 完工）；outputCmd
// 的旗写保留——兼任计算链的消耗落点（不写掉 a，循环会被优化掉瞬间完成），
// 真 kernel 接入时再决定是否恢复完工分诊用途。
// 独立成文件而非住进 prefill.cpp：prefill 是纯组装（g++ 编译，编不了
// <<<>>>），发射得住 .cu；do_dummy 原型在 all.cuh 供组装件调用。
// ===========================================================================
#include "all.cuh"

namespace kern {

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

// do_dummy —— host 发射小入口：全 SM 铺满（每 SM 8 block × 256 线程）。
// 流与旗字由组装件（prefill.cpp）传入
void do_dummy(cudaStream_t stream, volatile int* outputCmd, int outValue) {
    static int n_sm = 0;    // 一次性查 SM 数
    if (n_sm == 0) {
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
    }
    dummy_kernel<<<n_sm * 8, 256, 0, stream>>>(kLoadIters, outputCmd, outValue);
    check_launch("dummy");
}

}  // namespace kern
