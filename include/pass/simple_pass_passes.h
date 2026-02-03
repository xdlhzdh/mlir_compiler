#ifndef SIMPLE_PASS_PASSES_H
#define SIMPLE_PASS_PASSES_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct SimplePeepholePass : public PassInfoMixin<SimplePeepholePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
struct RemoveEmptyBlockPass : public PassInfoMixin<RemoveEmptyBlockPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
} // namespace llvm

#endif
