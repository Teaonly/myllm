# engine/Makefile —— 编译引擎：g++（纯 C++ 侧）+ nvcc（CUDA 侧）
#
# - vt 是单头组件：-I. 后以 "opt/vt/vt.hpp" 引用，零链接依赖
# - vt_cuda.cpp 含 CUDA kernel，用 nvcc -x cu 编译（文件名保持 .cpp）
# - 产物统一放 build/（根 .gitignore 已忽略），源码目录保持干净
# - CUDA 工具链在 /usr/local/cuda（本机 13.3，A100 机同路径）

CUDA_HOME ?= /usr/local/cuda
CXX       := g++
NVCC      := $(CUDA_HOME)/bin/nvcc

CXXFLAGS  := -std=c++17 -O0 -g -Wall -I. -I$(CUDA_HOME)/include
NVCCFLAGS := -std=c++17 -O0 -g -I. -x cu -Xcompiler -Wall

BIN  := build/engine
SRCS := main.cpp                     # g++ 侧源文件，往这列里加
OBJS := $(patsubst %.cpp,build/%.o,$(SRCS)) build/vt_cuda.o
DEPS := $(patsubst %.cpp,build/%.d,$(SRCS))

$(BIN): $(OBJS)
	$(CXX) -o $@ $^ -L$(CUDA_HOME)/lib64 -lcudart

build/%.o: %.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# vt_cuda.o：nvcc 专门规则（显式规则优先于上面的 %.o 模式），依赖手写（不走 -MMD）
build/vt_cuda.o: vt_cuda.cpp vt_cuda.hpp opt/vt/vt.hpp
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf build

-include $(DEPS)

.PHONY: clean
