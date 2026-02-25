#ifndef REMOVE_TRIVIAL_BLOCK_PASSES_H
#define REMOVE_TRIVIAL_BLOCK_PASSES_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct RemoveTrivialBlockPass : public PassInfoMixin<RemoveTrivialBlockPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
} // namespace llvm

#endif
