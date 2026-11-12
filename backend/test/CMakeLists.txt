# scheduler 重构中：旧测试（scheduler_test / engine_test）随实现一并移除，
# 历史见 git。新实现的第一步测试已重建——host 侧单测登记在 backend/Makefile
# 的 `test` 目标（make test，被测源码直接编入）。CMake 目标待模块成形、需要
# 独立构建树时再恢复，届时在此重新登记。
