cmake_minimum_required(VERSION 3.17)
project(myllm_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# vendored 单头库（nlohmann/json v3.11.3，仅 protocol.cpp 引用）
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opt)

# 引擎根目录进 include 路径：cuda/ 子目录里的头（cuda_engine.h 等）回头
# 引根目录的 engine.h / protocol.h 时，按文件所在目录找不到，须靠这条。
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 引擎进程：进程主体（main / protocol / scheduler）在根目录；CUDA 的具体
# 实现全部在 cuda/ 子目录（CudaEngine 当前为骨架，CUDA 落地后填充）。
add_executable(myllm_engine main.cpp protocol.cpp scheduler.cpp cuda/cuda_engine.cpp cuda/cache.cpp)
target_compile_options(myllm_engine PRIVATE -Wall -Wextra)

# cuda_runtime.h 的 include 路径与 cudart 链接由 CUDAToolkit 提供
# （CMake ≥3.17；编译机须装 CUDA toolkit；本机无 CUDA/编译器，不在此配置）
find_package(CUDAToolkit REQUIRED)
target_link_libraries(myllm_engine PRIVATE CUDA::cudart)

enable_testing()
add_subdirectory(test)
