# backend/Makefile —— 编译引擎：g++（C++ 侧）+ gcc（C 库）+ nvcc（CUDA 侧）
#
# - vt/ 是自包含模块：vt.hpp（DSL 机）+ vt_cuda.*（CUDA 绑定）+ vt_gguf.*（GGUF 加载）
# - vt_cuda.cpp 含 CUDA kernel，用 nvcc -x cu 编译（文件名保持 .cpp）
# - opt/gguf 是 antirez 的 C 库（GGUF 解析），gcc 直接编
# - 产物统一放 build/（根 .gitignore 已忽略），源码目录保持干净
# - CUDA 工具链在 /usr/local/cuda（本机 13.3，A100 机同路径）

CUDA_HOME ?= /usr/local/cuda
CXX       := g++
CC        := gcc
NVCC      := $(CUDA_HOME)/bin/nvcc

CXXFLAGS  := -std=c++17 -O0 -g -Wall -I. -Iopt -I$(CUDA_HOME)/include
CFLAGS    := -O2 -Wall
NVCCFLAGS := -std=c++17 -O0 -g -I. -x cu -Xcompiler -Wall

BIN   := build/backend
SRCS  := main.cpp scheduler.cpp protocol.cpp vt/vt_gguf.cpp  # g++ 侧源文件，往这列里加
CSRCS := opt/gguf/gguflib.c opt/gguf/fp16.c      # C 库
OBJS  := $(patsubst %.cpp,build/%.o,$(SRCS)) build/vt/vt_cuda.o \
         $(patsubst %.c,build/%.o,$(CSRCS))
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

clean:
	rm -rf build

# host 侧单元测试（无 GPU / 无 CUDA；被测源码直接编入测试目标——库目标待
# 模块成形再拆）。跑法：make test
build/scheduler_test: test/scheduler_test.cpp scheduler.cpp protocol.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: build/scheduler_test
	build/scheduler_test

-include $(DEPS)

.PHONY: clean test
