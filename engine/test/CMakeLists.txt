# host 侧单元测试（无 GPU / 无 CUDA 依赖；device 侧测试回 CUDA 机另列）。
# 被测源码直接编入测试目标——库目标（scheduler / cache / main 装配）待模块成形再拆。

add_executable(scheduler_test scheduler_test.cpp ../scheduler.cpp)
target_include_directories(scheduler_test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_options(scheduler_test PRIVATE -Wall -Wextra)
add_test(NAME scheduler COMMAND scheduler_test)

add_executable(engine_test engine_test.cpp ../engine.cpp ../protocol.cpp ../scheduler.cpp)
target_include_directories(engine_test PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_options(engine_test PRIVATE -Wall -Wextra)
add_test(NAME engine COMMAND engine_test)
