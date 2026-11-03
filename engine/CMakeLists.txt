cmake_minimum_required(VERSION 3.16)
project(myllm_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# vendored 单头库（nlohmann/json v3.11.3，仅 protocol.cpp 引用）
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opt)

# 引擎进程（executor 为占位实现，CUDA 落地后替换）
add_executable(myllm_engine main.cpp engine.cpp protocol.cpp scheduler.cpp)
target_compile_options(myllm_engine PRIVATE -Wall -Wextra)

enable_testing()
add_subdirectory(test)
