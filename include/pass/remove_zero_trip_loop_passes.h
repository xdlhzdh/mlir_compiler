#ifndef REMOVE_ZERO_TRIP_LOOP_PASSES_H
#define REMOVE_ZERO_TRIP_LOOP_PASSES_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct RemoveZeroTripLoopPass : public PassInfoMixin<RemoveZeroTripLoopPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
} // namespace llvm

#endif
