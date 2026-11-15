# backend/Makefile —— 编译引擎：g++（C++ 侧）+ gcc（C 库）+ nvcc（CUDA 侧）
#
# - vt/ 是自包含模块：vt.hpp（DSL 机）+ vt_cuda.*（CUDA 绑定）+ vt_gguf.*（GGUF 加载）
# - vt_cuda.cpp 用 nvcc -x cu 编译（文件名保持 .cpp）
# - kernel/ 是 Kernel 开发目录：prefill.cpp 走 nvcc -x cu（要发射 kernel，
#   g++ 编不了 <<<>>>，同 vt_cuda.cpp 路数）；接口头 index.h 两侧共用
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
NVCCFLAGS := -std=c++17 -O0 -g -arch=sm_80 -I. -x cu -Xcompiler -Wall

BIN   := build/backend
SRCS  := main.cpp scheduler.cpp protocol.cpp cache.cpp vt/vt_gguf.cpp  # g++ 侧源文件，往这列里加
CSRCS := opt/gguf/gguflib.c opt/gguf/fp16.c      # C 库
OBJS  := $(patsubst %.cpp,build/%.o,$(SRCS)) build/vt/vt_cuda.o \
         build/kernel/prefill.o build/kernel/embed.o $(patsubst %.c,build/%.o,$(CSRCS))
DEPS  := $(OBJS:.o=.d)

$(BIN): $(OBJS)
	$(CXX) -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart

build/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

build/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# vt_cuda.o：nvcc 专门规则（显式规则优先于 %.o 模式），依赖手写（不走 -MMD）
build/vt/vt_cuda.o: vt/vt_cuda.cpp vt/vt_cuda.hpp vt/vt.hpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# kernel/prefill.o、kernel/embed.o：同上——nvcc -x cu，头文件依赖手写
build/kernel/prefill.o: kernel/prefill.cpp kernel/index.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/kernel/embed.o: kernel/embed.cpp kernel/index.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

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
