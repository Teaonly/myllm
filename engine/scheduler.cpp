// ===========================================================================
// scheduler.cpp — 空实现（重构起点，见 scheduler.h 头注）
// ===========================================================================
#include "scheduler.h"

namespace myllm {

Scheduler::Scheduler(const SchedulerConfig& cfg, std::istream& in, std::ostream& out)
    : cfg_(cfg), 
      in_(in), 
      out_(out) {}




}  // namespace myllm
