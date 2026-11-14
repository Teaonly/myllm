// ===========================================================================
// index.cuh — kernel 对外接口：全部是裸指针 + 形状参数，无 CUDA 类型
// ===========================================================================

#pragma once

#include <cstdint>
#include "vt/vt_cuda.hpp"

namespace kern {

// 同步词一律 volatile int*：数据由对端（kernel/host）经 PCIe 直写 mapped
// 锁页内存，任何一侧都不许把值缓进寄存器
void do_prefill_chunking(CudaEnv* env, volatile int* left, volatile int* right);
void do_prefill_chunked(CudaEnv* env, volatile int* left, volatile int* right);

}  // namespace kern
