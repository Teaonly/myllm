# backend/Makefile —— 编译引擎：g++（C++ 侧）+ gcc（C 库）+ nvcc（CUDA 侧）
#
# - vt/ 是自包含模块：vt.hpp（DSL 机）+ vt_cuda.*（CUDA 绑定）+ vt_gguf.*（GGUF 加载）
# - vt_cuda.cpp 用 nvcc -x cu 编译（文件名保持 .cpp——.cu 后缀留给 kernel/）
# - kernel/ 是 Kernel 开发目录，组织规矩（两层入口 + 组装）：
#     index.h   对外大入口（do_prefill_*，一整段计算的组装序列）：
#               裸指针 + 形状参数，g++/nvcc 两侧共用，目录外只认它
#     all.cuh   内部小入口的声明集（do_* 小步 + 设备侧原型/类型，设备段
#               锁 __CUDACC__ 供 g++ 组装件 include）：只有声明没有实现
#     *.cu      实现件：一主题一文件（kernel 实现 + do_* 发射器，
#               embed/dummy/dq/...），跨主题复用的设备函数独立成文件
#               （dq.cu）；新增 .cu 自动进编译（wildcard 收集）
#     prefill.cpp 组装件（g++ 侧）：大入口的实现，只按序调 do_* 小入口
# - opt/gguf 是 antirez 的 C 库（GGUF 解析），gcc 直接编
# - 产物统一放 build/（根 .gitignore 已忽略），源码目录保持干净
# - CUDA 工具链在 /usr/local/cuda（本机 13.0，A100 机同路径）

CUDA_HOME ?= /usr/local/cuda
CXX       := g++
CC        := gcc
NVCC      := $(CUDA_HOME)/bin/nvcc

CXXFLAGS  := -std=c++17 -O0 -g -Wall -I. -Iopt -I$(CUDA_HOME)/include
CFLAGS    := -O2 -Wall
# -arch=sm_80：本机 CMP 170HX 与目标 A100 同为 GA100（compute 8.0）
NVCCFLAGS := -std=c++17 -O0 -g -arch=sm_80 -I. -Xcompiler -Wall

BIN   := build/backend
SRCS  := main.cpp scheduler.cpp protocol.cpp cache.cpp vt/vt_gguf.cpp \
         kernel/prefill.cpp   # g++ 侧源文件，往这列里加（prefill.cpp=kernel/ 组装件）
CSRCS := opt/gguf/gguflib.c opt/gguf/fp16.c      # C 库
KRNLS := $(patsubst %.cu,build/%.o,$(wildcard kernel/*.cu))  # kernel/*.cu 全收
DLNK  := build/kernel/dlink.o                    # 设备链接块（-rdc 收尾）
OBJS  := $(patsubst %.cpp,build/%.o,$(SRCS)) build/vt/vt_cuda.o $(KRNLS) $(DLNK) \
         $(patsubst %.c,build/%.o,$(CSRCS))
DEPS  := $(OBJS:.o=.d)

$(BIN): $(OBJS)
	$(CXX) -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart -lcudadevrt

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# vt_cuda.o：.cpp 伪装 CUDA，-x cu 强制；显式规则优先于 %.o 模式，
# 头文件依赖手写（nvcc 不走 -MMD）
build/vt/vt_cuda.o: vt/vt_cuda.cpp vt/vt_cuda.hpp vt/vt.hpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

# kernel/*.cu：真 CUDA 源（后缀即身份，不用 -x cu）。公共头两件：all.cuh
# （声明集）+ index.h（对外接口），qwen3.8-next.h 是几何常量头（embed 用
# 到；宁多列勿漏），依赖手写，一条模式规则统一。-rdc=true：__device__ 跨
# .cu 调用（embed_rows_kernel → dq.cu 的 dq_q6_k）要可重定位设备码，随后
# 由 dlink 步合并（__global__ 跨 .cu 发射本不需要，坐一趟车无妨）
build/kernel/%.o: kernel/%.cu kernel/all.cuh kernel/index.h qwen3.8-next.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -rdc=true -c $< -o $@

# 设备链接块：-rdc 的各 kernel 对象设备码在此合并（跨文件 __device__ 调用
# 的符号在这步解决），随 $(BIN) 进最终 g++ 链接
build/kernel/dlink.o: $(KRNLS)
	$(NVCC) -arch=sm_80 -dlink $^ -o $@

clean:
	rm -rf build

FEED_HEAD ?= 30
FEED_GAP  ?= 0.2
FEED_TAIL ?= 5
test_single: $(BIN)
	./test/feed.sh test/single.ndjson $(FEED_HEAD) $(FEED_GAP) $(FEED_TAIL) | $(BIN); \
	st=$$?; case $$st in 1|254) ;; *) exit $$st ;; esac

-include $(DEPS)

.PHONY: clean test_single
