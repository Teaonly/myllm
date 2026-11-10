#pragma once

// vt_gguf —— GGUF 文件加载层：把张量的原始字节搬进 CudaTensor 设备显存。
//
// "原始内容" = 字节级搬运，不做任何 dtype 转换（F32/BF16/Q4...怎么解释
// 是消费词的事）；张量元数据（type/dim）也不进 CudaTensor——形状的真相
// 在模型常量/剧本里，显存块只管字节。
//
// 词表：gguf.load ( path name -- tensor )  逐张量重开文件（mmap 很廉价），
// 批量加载将来再上 gguf.open 常驻句柄。

#include "vt_cuda.hpp"

void register_gguf_words(CudaEnv& env);
