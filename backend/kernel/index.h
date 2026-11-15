// ===========================================================================
// index.cuh — kernel 对外接口：全部是裸指针 + 形状参数，无 CUDA 类型
// ===========================================================================

#pragma once

#include <cstdint>
#include "vt/vt_cuda.hpp"
#include "scheduler.h"

namespace kern {

// 同步词一律 volatile int*：数据由对端（kernel/host）经 PCIe 直写 mapped
// 锁页内存，任何一侧都不许把值缓进寄存器
void do_prefill_chunking(CudaEnv* env, myllm::SchedulerStaff& staff);
void do_prefill_chunked(CudaEnv* env, myllm::SchedulerStaff& staff);

// embedding gather（kernel/embed.cpp）：窗口 [inputLeft,inputRight) 取
// min(CS, 余量) 个 token，Q6_K 词表行解进 pf.emb，广播进 pf.R (T,HC,N)。
// 调试落盘：MYLLM_EMBED_DUMP=<path> 同步写 pf.emb 前 T 行裸 f32
void do_embed(CudaEnv* env, myllm::SchedulerStaff& staff);

}  // namespace kern
