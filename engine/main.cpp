// ===========================================================================
// main.cpp — 进程入口：stdin/stdout 装配（协议 codec 在 Engine 内部）
// ===========================================================================
// executor 当前为占位实现：无 GPU 环境跑通协议链路用（每请求 5,6,EOS 三拍
// 后自然收尾）。CUDA 合并步 executor（scheduler.md §3/S8 事件链）落地后
// 整体替换，Engine 与 Scheduler 不动。
// ===========================================================================
#include <iostream>
#include <map>
#include <vector>

#include "engine.h"
#include "model_constants.h"

namespace {

class StubExecutor final : public engine::StepExecutor {
public:
    void run_step(const std::vector<engine::Segment>& segs, engine::StepResult& out) override {
        out.per_lane.assign(2, engine::LaneResult{});
        for (const auto& g : segs) {
            if (!g.samples) continue;
            int32_t t;
            switch (++count_[g.uid]) {
                case 1: t = 5; break;
                case 2: t = 6; break;
                default: t = model::EOS; break;
            }
            out.per_lane[g.slot] = {true, t};
        }
    }

private:
    std::map<int, int> count_;
};

}  // namespace

int main() {
    StubExecutor exec;
    engine::Engine eng(engine::SchedulerConfig{}, exec, std::cin, std::cout);
    eng.run();
    return 0;
}
