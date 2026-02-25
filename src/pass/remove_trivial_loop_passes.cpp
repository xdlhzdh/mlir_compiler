// RemoveTrivialLoopPass：删除“死循环”（无 exit 的 loop）。
//
// 目标 IR 形态示例：
//   entry:
//     br label %loop
//   loop:
//     br label %loop    ; 无 exit，永远不返回
//
// 变换后：entry 的 terminator 改为 unreachable，loop 块被
// removeUnreachableBlocks 删除。
//
// 安全约束：仅当“进入 loop 的所有外部前驱”都是无条件直接跳到 header
// 时，才改写。 若存在条件分支（如 br i1 %cond, label %loop, label
// %exit），则跳过，避免误删 %exit 可达路径。

#include "pass/remove_trivial_loop_passes.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

PreservedAnalyses RemoveTrivialLoopPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {

  // 获取 LoopInfo 和 DominatorTree，用于识别 loop 结构并增量更新支配树
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  // Lazy 策略：批量收集 CFG 变更后再统一更新 DT，避免频繁重算
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);

  bool Changed = false;

  // 复制 loop 列表，因为后续会修改 CFG，迭代器可能失效
  SmallVector<Loop *, 8> Loops(LI.begin(), LI.end());

  for (Loop *L : Loops) {

    // 只处理最内层 loop（无子 loop），简化逻辑
    if (!L->getSubLoops().empty())
      continue;

    // 只处理“死循环”：无 exit 边，即 getExitBlock() 为 null
    // 有 exit 的 loop 不删，保留正常控制流
    if (L->getExitBlock())
      continue;

    // Header：循环的入口基本块，loop 外的前驱和 loop 内的回边都指向它。
    // 例：entry -> loop (header) <- loop (回边)
    //     entry:
    //       br label %loop
    //     loop:              ; Header，有前驱 entry（外）和 loop（回边）
    //       br label %loop
    BasicBlock *Header = L->getHeader();

    // 收集所有“来自 loop 外”的 header 前驱
    // 这些块是进入死循环的入口，将被改写为 unreachable
    SmallVector<BasicBlock *, 4> OutsidePreds;
    for (BasicBlock *Pred : predecessors(Header)) {
      if (!L->contains(Pred))
        OutsidePreds.push_back(Pred);
    }

    // 安全检查：仅当所有外部前驱都是“无条件 br 到 Header”时才改写
    // 若任一前驱是条件分支（如 br i1 %c, label %loop, label %exit），
    // 则不能把该 terminator 改成 unreachable，否则会误删 %exit 可达路径
    bool CanRewrite = true;
    for (BasicBlock *Pred : OutsidePreds) {
      auto *Br = dyn_cast<BranchInst>(Pred->getTerminator());
      if (!Br || !Br->isUnconditional() || Br->getSuccessor(0) != Header) {
        CanRewrite = false;
        break;
      }
    }
    if (!CanRewrite)
      continue;

    // 将每个外部前驱的 terminator 改为 unreachable
    // PreserveLCSSA=true 保持 LCSSA 形式；DTU 用于增量更新支配树
    for (BasicBlock *Pred : OutsidePreds)
      changeToUnreachable(Pred->getTerminator(), /*PreserveLCSSA*/ true, &DTU);

    DTU.flush();
    // 入口边已 unreachable，loop 内所有块变为不可达，统一删除
    removeUnreachableBlocks(F, &DTU);

    Changed = true;
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // CFG 已变，LoopInfo 失效，不 preserve；仅保留 DominatorTree（已通过 DTU
  // 更新）
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();

  return PA;
}
