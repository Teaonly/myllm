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

# ---- 回放测试 target：剧本驱动引擎，无需前端（test/case.ndjson）----
#
# 两个 target 都是"假前端"：stdin 喂剧本、协议 stdout 打到终端、stderr 打
# 调试信息。为什么必须握住 stdin：EOF 在本引擎 = 前端死亡 = 弃在途工作，
# 剧本一抽干下一拍就是 EOF——所以两个 target 都在剧本后保持 stdin 打开一
# 段时间（排空语义落地前的过渡方案，见 run_loop 头注）。
#   rw（①窗口法，replay-window）：剧本后保持 WINDOW 秒再关——测 EOF
#     路径；窗口就是引擎的生成时间。
#   rh（②常开法，replay-hold）：剧本后自动补一条 ExitMsg、再保持 HOLD
#     秒——测优雅退出路径（管道常开，ExitMsg 不与 EOF 同拍竞态）。
# 不用 tail -f /dev/null 做永驻：引擎退出后管道左侧还活着，make 会永远
# 等它；有界 sleep 语义相同且 target 必然返回。
# 秒数可配：cmake -DREPLAY_WINDOW=10 -DREPLAY_HOLD=5 ..
find_program(BASH_EXE bash REQUIRED)

set(REPLAY_WINDOW 5 CACHE STRING "rw：剧本后 stdin 保持秒数")
set(REPLAY_HOLD 3 CACHE STRING "rh：ExitMsg 后 stdin 保持秒数")

add_custom_target(rw
  COMMAND ${BASH_EXE} -c [[{ cat "$1"; sleep "$3"; } | "$2"]] rw
          ${CMAKE_CURRENT_SOURCE_DIR}/test/case.ndjson
          $<TARGET_FILE:myllm_engine> ${REPLAY_WINDOW}
  DEPENDS myllm_engine
  COMMENT "回放 case.ndjson（窗口 ${REPLAY_WINDOW}s 后 EOF）"
  VERBATIM)

add_custom_target(rh
  COMMAND ${BASH_EXE} -c
          [[{ cat "$1"; printf '%s\n' '{"__type__":"ExitMsg"}'; sleep "$3"; } | "$2"]]
          rh
          ${CMAKE_CURRENT_SOURCE_DIR}/test/case.ndjson
          $<TARGET_FILE:myllm_engine> ${REPLAY_HOLD}
  DEPENDS myllm_engine
  COMMENT "回放 case.ndjson + ExitMsg（常开 ${REPLAY_HOLD}s，优雅退出）"
  VERBATIM)
