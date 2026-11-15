// ===========================================================================
// index.h — kernel 目录对外接口：大入口（do_prefill_*），目录外只认这一件头
// ===========================================================================
// 两层入口的分工：本文件是给目录外（scheduler 等）用的大入口——一个大
// 入口是一整段计算的组装序列（实现 kernel/prefill.cpp，纯组装）；kernel/
// 内部的小入口（do_embed/do_dummy、kernel 原型）在 all.cuh，目录外不认。
// 大入口只收 env + staff 两个组装材料箱（张量指针/流/同步词由组装件自己
// 现取），调用方只管拍子——所以两个参数就够。
// ===========================================================================
#pragma once

#include "vt/vt_cuda.hpp"    // CudaEnv
#include "scheduler.h"       // myllm::SchedulerStaff

namespace kern {

// prefill 两条路径（kernel/prefill.cpp 组装）：一拍 = 窗口内 token 的
// embedding（do_embed，真计算）+ 余下层栈（当前 do_dummy 占位，真 kernel
// 逐步接入后组装序列随之长）。分块喂（CHUNKING）与末块（CHUNKED）两拍
// 同路，行为差在组装内部（dummy 分诊值 1/2）
void do_prefill_chunking(CudaEnv* env, myllm::SchedulerStaff& staff);
void do_prefill_chunked(CudaEnv* env, myllm::SchedulerStaff& staff);

void reset_session(CudaEnv* env, myllm::SchedulerStaff& staff, target);

}  // namespace kern
