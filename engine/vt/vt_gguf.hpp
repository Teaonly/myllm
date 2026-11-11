#pragma once

// vt_gguf —— GGUF 文件加载层：把张量的原始字节搬进 CudaTensor 设备显存。
//
// "原始内容" = 字节级搬运，不做任何 dtype 转换（F32/BF16/Q4...怎么解释
// 是消费词的事）；张量元数据（type/dim）也不进 CudaTensor——形状的真相
// 在模型常量/剧本里，显存块只管字节。
//
// 词表：gguf.load  ( path name dst -- dst )  原始字节拷进剧本给定的 dst
//       （dev=异步 H2D / host=memcpy 直达；不自建 Tensor，先 probe 探尺寸
//        再 cuda.alloc / host.alloc，落点与 slab 切片由剧本决定）
//       gguf.probe ( path name -- n )      只读探针：张量字节数，不存在即 panic。
// 批量加载将来再上 gguf.open 常驻句柄。

#include "vt_cuda.hpp"

void register_gguf_words(CudaEnv& env);
