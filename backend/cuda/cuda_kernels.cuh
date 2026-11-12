// ===========================================================================
// cuda_kernels.cuh — kernel 对外接口：全部是裸指针 + 形状参数，无 CUDA 类型
// ===========================================================================
//
// Kernel 开发阶段的接口约定（backend/cuda 目录）：
//   - 本头被两侧共用：g++ 侧（ktest.cpp 对拍驱动 / 未来的 forward 驱动）与
//     nvcc 侧（各 .cu 实现）。所以签名里不出现 CUDA 类型——流用 void*，
//     bf16 像用 uint16_t*（bit 即 __nv_bfloat16 的位型），避免 g++ 侧被迫
//     include CUDA 头；
//   - 数值契约跟模型常量头（qwen3.8-next.h）：激活与权重 bf16 进 f32 算、
//     递归状态 f32；本层 kernel 一律 f32 进出，bf16 像由调用方按需另取
//     （GEMM 输入 / 参考契约落盘点）；
//   - 排布一律行主序；(rows, width) 的 width 是最后一维。kernel 只认
//     连续段，batch/头维由调用方打平；
//   - 无返回值：参数非法直接 fail fast（fprintf + abort，同 vt_panic 风格
//     但不依赖 vt 栈——kernel 层可独立于 vt 测试）。
// ===========================================================================
#pragma once

#include <cstdint>

namespace kern {

// 分组 RMSNorm（qwen4_exp modeling:Qwen4ExpTextRMSNorm，group_size 特例）：
//   y[i] = x[i] * rsqrt(mean_{段内}(x²) + eps) * (1 + w[i])
//   x 按 (rows, width) 打平，width 必须是 group 的整数倍；每 group 个连续
//   元素为一段独立归一（hc_norm：width=10240/group=2560，即残差流 4 条各
//   自归一；q_norm/k_norm：width=group=256；indexer 两 norm：128）。
//   w 可为 nullptr（乘 1）；y16 可为 nullptr（不写 bf16 像，否则同值另写
//   一份——pf.xn16 一类的 GEMM 入参/落盘像，与 y 内容一致仅位宽不同）。
//   eps 进 sqrt（模型 RMS_EPS=1e-6）。w 按 (width,) 逐通道。
void rmsnorm(const float* x, const float* w, float* y, uint16_t* y16,
             int rows, int width, int group, float eps, void* stream);

}  // namespace kern
