#include "pass/simple_pass_passes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;
using namespace llvm::PatternMatch;

PreservedAnalyses SimplePeepholePass::run(Function &F, FunctionAnalysisManager &) {
  bool changed = false;
  for (auto &BB : F) {
    for (auto It = BB.begin(); It != BB.end();) {
      Instruction *I = &*It++;
      Value *X;
      if (match(I, m_Add(m_Value(X), m_Zero()))) {
        I->replaceAllUsesWith(X);
        I->eraseFromParent();
        changed = true;
      }
    }
  }
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

PreservedAnalyses RemoveEmptyBlockPass::run(Function &F, FunctionAnalysisManager &) {
  bool changed = false;
  for (auto It = F.begin(); It != F.end();) {
    BasicBlock &BB = *It++;
    if (&BB == &F.getEntryBlock())
      continue;
    if (BB.size() != 1)
      continue;
    auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
    if (!Br || !Br->isUnconditional())
      continue;
    BasicBlock *Succ = Br->getSuccessor(0);
    BB.replaceAllUsesWith(Succ);
    BB.eraseFromParent();
    changed = true;
  }
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
