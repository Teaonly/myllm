cmake_minimum_required(VERSION 3.17)
project(myllm_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# vendored 单头库（nlohmann/json v3.11.3，仅 protocol.cpp 引用）
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opt)

# 引擎进程（CudaEngine 为骨架实现，CUDA 落地后填充）
add_executable(myllm_engine main.cpp protocol.cpp scheduler.cpp cuda_engine.cpp)
target_compile_options(myllm_engine PRIVATE -Wall -Wextra)

# cuda_runtime.h 的 include 路径与 cudart 链接由 CUDAToolkit 提供
# （CMake ≥3.17；编译机须装 CUDA toolkit；本机无 CUDA/编译器，不在此配置）
find_package(CUDAToolkit REQUIRED)
target_link_libraries(myllm_engine PRIVATE CUDA::cudart)

enable_testing()
add_subdirectory(test)
