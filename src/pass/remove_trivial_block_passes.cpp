#include "pass/remove_trivial_block_passes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses RemoveTrivialBlockPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {

  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  bool Changed = false;

  SmallVector<DominatorTree::UpdateType, 8> DTUpdates;

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

    // 🔹 更新 PHI
    Succ->removePredecessor(&BB);

    // 🔹 记录 DT 更新
    for (auto *Pred : predecessors(&BB)) {
      DTUpdates.push_back({DominatorTree::Delete, Pred, &BB});
      DTUpdates.push_back({DominatorTree::Insert, Pred, Succ});
    }

    BB.replaceAllUsesWith(Succ);
    BB.eraseFromParent();

    Changed = true;
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // 🔹 增量更新 DominatorTree
  DT.applyUpdates(DTUpdates);

  // LoopInfo 因 CFG 变更已失效，不 preserve 以便后续 pass 重新计算
  // PA的缺省值是None, 意味着缺省全部不保留
  // 返回的PA会立马被 AM.invalidate(F, PA)调用
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();

  return PA;
}
