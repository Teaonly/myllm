// ===========================================================================
// index.cuh — kernel 对外接口：全部是裸指针 + 形状参数，无 CUDA 类型
// ===========================================================================

#pragma once

#include <cstdint>
#include "vt/vt_cuda.hpp"

namespace kern {

void do_prefill(CudaEnv* env, const int* left, const int* right);

}  // namespace kern
