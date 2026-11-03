// ===========================================================================
// scheduler.cpp — 空实现（重构起点，见 scheduler.h 头注）
// ===========================================================================
#include "scheduler.h"

namespace engine {

Scheduler::Scheduler(const SchedulerConfig& cfg, StepExecutor& exec, OutputSink& sink)
    : cfg_(cfg), exec_(exec), sink_(sink) {}

void Scheduler::admit(int, std::vector<int32_t>, const proto::SamplingParams&) {}
void Scheduler::abort(int) {}
void Scheduler::request_exit() {}
void Scheduler::pump() {}
bool Scheduler::idle() const { return true; }

}  // namespace engine
