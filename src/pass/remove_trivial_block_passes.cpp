#include "pass/remove_trivial_block_passes.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

PreservedAnalyses RemoveTrivialBlockPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {

  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  // 用 DomTreeUpdater(Lazy) 自动管理 BB 删除与 DT edge 更新的顺序，
  // 避免在新版本 LLVM 下手动 applyUpdates 时违反"被删节点必须是叶子"的不变量。
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  bool Changed = false;

  // 收集候选块，避免边迭代边修改 F 的 BasicBlock 列表
  SmallVector<BasicBlock *, 8> Candidates;
  for (BasicBlock &BB : F) {
    if (&BB == &F.getEntryBlock())
      continue;
    if (BB.size() != 1)
      continue;
    auto *Br = dyn_cast<BranchInst>(BB.getTerminator());
    if (!Br || !Br->isUnconditional())
      continue;
    Candidates.push_back(&BB);
  }

  for (BasicBlock *BB : Candidates) {
    auto *Br = cast<BranchInst>(BB->getTerminator());
    BasicBlock *Succ = Br->getSuccessor(0);

    // 收集前驱（提前快照，下面 setSuccessor 会改变前驱集合）
    SmallVector<BasicBlock *, 4> Preds(predecessors(BB));

    // 把每个前驱的 BB 后继重定向到 Succ；同步更新 PHI 与 DTU
    SmallVector<DominatorTree::UpdateType, 8> Updates;
    for (BasicBlock *Pred : Preds) {
      Instruction *TI = Pred->getTerminator();
      for (unsigned i = 0, e = TI->getNumSuccessors(); i < e; ++i) {
        if (TI->getSuccessor(i) == BB)
          TI->setSuccessor(i, Succ);
      }
      Updates.push_back({DominatorTree::Insert, Pred, Succ});
      Updates.push_back({DominatorTree::Delete, Pred, BB});
    }

    // 修复 Succ 的 PHI：把来自 BB 的 incoming 改为来自每个 Pred
    for (PHINode &PHI : Succ->phis()) {
      Value *V = PHI.getIncomingValueForBlock(BB);
      PHI.removeIncomingValue(BB, /*DeletePHIIfEmpty=*/false);
      for (BasicBlock *Pred : Preds)
        PHI.addIncoming(V, Pred);
    }

    Updates.push_back({DominatorTree::Delete, BB, Succ});
    DTU.applyUpdates(Updates);

    // 现在 BB 已经从 CFG 中孤立；交给 DTU 删除
    DTU.deleteBB(BB);
    Changed = true;
  }

  if (!Changed)
    return PreservedAnalyses::all();

  DTU.flush();

  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
