#ifndef REMOVE_TRIVIAL_LOOP_PASSES_H
#define REMOVE_TRIVIAL_LOOP_PASSES_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct RemoveTrivialLoopPass : public PassInfoMixin<RemoveTrivialLoopPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
} // namespace llvm

#endif
