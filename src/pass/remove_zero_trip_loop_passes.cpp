// ============================================================================
// RemoveZeroTripLoopPass — 删除 trip count = 0 的循环（循环体永不执行）
// ============================================================================
//
// 概念：backedge taken count (BTC)
//   SE 计算的"回边被执行的次数"。trip count = BTC + 1。
//   当 BTC = 0 时，回边从未执行，即循环体一次都不会运行。
//
// 算法步骤：
//   1. 遍历所有最内层循环，通过 ScalarEvolution 检查 BTC 是否为常量 0
//   2. 将 Exit 块中对循环内定义值的引用替换为"进入循环前的初值"
//   3. 将所有"循环外 -> Header"的 CFG 边改写为"循环外 -> Exit"
//   4. 修复 Exit 块的 PHI 节点（移除 Header incoming，补上 Preheader incoming）
//   5. 删除不可达的循环块
//
// ============================================================================

#include "pass/remove_zero_trip_loop_passes.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

PreservedAnalyses RemoveZeroTripLoopPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  // --- 获取本 pass 依赖的三个分析结果 ---
  auto &LI = AM.getResult<LoopAnalysis>(F); // 循环结构信息
  auto &SE =
      AM.getResult<ScalarEvolutionAnalysis>(F); // 标量演化（推导循环计数）
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F); // 支配树

  // Lazy 策略：批量 CFG 修改后再统一刷新支配树，减少中间更新开销
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
  bool Changed = false;

  // 先复制循环列表：后续删除循环块会使 LoopInfo 迭代器失效
  SmallVector<Loop *, 8> Loops(LI.begin(), LI.end());

  // 注意：Loop 对象只包含循环体内部的块，Preheader 和 Exit 不属于 Loop。
  // L->contains(BB) 的返回值（对照 remove_zero_trip_loop.ll）：
  //
  //   块              | L->contains() | 说明
  //   --------------- | ------------- | ----
  //   %entry (Preheader) | false      | 循环外，只是跳进 Header 的入口
  //   %loop  (Header)    | true       | 循环入口判定块，属于循环体
  //   %body  (Body/Latch)| true       | 循环体+回边块，属于循环体
  //   %exit  (Exit)      | false      | 循环外，循环结束后到达的块
  //
  for (Loop *L : Loops) {
    // 只处理最内层循环（无子循环），回避嵌套零 trip 的复杂情况
    if (!L->getSubLoops().empty())
      continue;

    // ------------------------------------------------------------------
    // 判定：是否为零 trip 循环
    // ------------------------------------------------------------------
    // API: SE.getBackedgeTakenCount(L)
    //   返回 SCEV 表达式，表示回边执行次数。若无法推导则返回
    //   SCEVCouldNotCompute。
    const SCEV *BTC = SE.getBackedgeTakenCount(L);
    if (isa<SCEVCouldNotCompute>(BTC))
      continue;

    // 只处理 BTC 为编译期常量 0 的情况
    const SCEVConstant *C = dyn_cast<SCEVConstant>(BTC);
    if (!C || !C->getAPInt().isZero())
      continue;

    // ------------------------------------------------------------------
    // 循环结构概念（结合 `remove_zero_trip_loop.ll` 示例）
    // ------------------------------------------------------------------
    //
    // 概念            | LLVM API                  | 示例中对应块
    // --------------- | ------------------------- | -----------
    // Preheader       | L->getLoopPreheader()     | %entry
    // Header          | L->getHeader()            | %loop
    // Body            | （循环内计算指令所在块）   | %body
    // Latch           | L->getLoopLatch()         | %body（与 Body 合并）
    // Exit            | L->getExitBlock()         | %exit
    //
    // Preheader（预头块）：循环外、唯一跳到 Header 的前驱。
    //   提供单一入口，便于提取初值、做 CFG 重写。
    //   若不存在（Header 有多个循环外前驱），本 pass 保守跳过。
    //
    // Header（循环头）：循环的入口判定块，所有进入循环的路径都先到这里。
    //   通常包含条件分支：满足条件进入 Body，否则跳到 Exit。
    //
    // Body（循环体）：真正执行计算/副作用的区域。
    //
    // Latch（回边块）：每次迭代末尾把控制流跳回 Header 的块。
    //   Body 和 Latch 可以分开，也可以合并为同一个基本块（如本例）。
    //
    // Exit（出口块）：循环结束后控制流到达的块。
    //
    // 对应 CFG：
    //   %entry ──> %loop ──> %body ──> %loop
    //                  \──────────────> %exit
    //
    // 本例中 `%cmp = icmp slt i32 %i, 0` 首轮即为假（0 < 0 = false），
    // 实际执行路径：%entry -> %loop -> %exit，%body 永不执行。
    //
    BasicBlock *Header = L->getHeader();
    BasicBlock *Exit = L->getExitBlock();
    BasicBlock *Preheader = L->getLoopPreheader();

    // 前置条件：要求单 Exit 块 + 存在 Preheader
    if (!Exit || !Preheader)
      continue;

    // 收集循环外的 Header 前驱（有 Preheader 时通常只有它一个）
    SmallVector<BasicBlock *, 4> OutsidePreds;
    for (BasicBlock *Pred : predecessors(Header)) {
      if (!L->contains(Pred))
        OutsidePreds.push_back(Pred);
    }

    // ------------------------------------------------------------------
    // 步骤 A：修正 Exit 对循环内定义值的引用
    // ------------------------------------------------------------------
    // 为什么要先做这步：
    //   后续步骤 B 会把 Preheader->Header 改写为 Preheader->Exit，
    //   步骤 C 会删除不可达的循环块。若 Exit 仍引用循环内定义的值，
    //   这些 use-def 链将悬空，导致 IR 非法。
    //
    // 处理策略：
    //   零 trip 时循环体未执行，Exit 中使用的循环内值应取"进入循环前的初值"。
    //   对 Header 的 PHI（典型归纳变量），其初值来自 Preheader 路径的
    //   incoming。
    //
    // 示例（对照 remove_zero_trip_loop.ll）：
    //   %loop:  %i = phi i32 [0, %entry], [%i.next, %body]
    //   %exit:  ret i32 %i    <-- 引用了循环内的 %i
    //   零 trip 时 %i 应等于 %entry 路径的初值 0，故改写为 ret i32 0
    //
    // LLVM API 说明（以 `ret i32 %i` 为例）：
    //
    //   `ret i32 %i` 这条指令有 1 个操作数：%i。
    //   LLVM 用 Use 对象来表示这种"谁用了谁"的关系：
    //
    //     ret 指令
    //       └─ 操作数 #0 (Use) ──引用──> %i (Value)
    //
    //   Use &U  — 就是上面这根"引用箭头"，它记录了：
    //             "ret 指令的第 0 个操作数位置，当前指向 %i"
    //   U.get() — 顺着箭头拿到被引用的值，即 %i
    //   U.set(新值) — 把箭头改指向别的值。
    //             例如 U.set(常量 0) 之后，IR 变成 `ret i32 0`
    //             不需要手动更新其他数据结构，LLVM 内部自动维护。
    //
    //   dyn_cast<T>(V) — 安全类型转换，失败返回 nullptr
    //   PHI.getIncomingValueForBlock(B) — 取 PHI 在前驱 B 对应的 incoming 值
    //
    BasicBlock *OutsidePred = Preheader;
    for (Instruction &I : *Exit) {
      for (Use &U : I.operands()) {
        Value *V = U.get();

        // 跳过非指令类型（Constant / Argument 等），它们不在循环内定义
        Instruction *Def = dyn_cast<Instruction>(V);
        if (!Def)
          continue;

        // 跳过定义点在循环外的值，它们删除循环后仍可用
        if (!L->contains(Def->getParent()))
          continue;

        // 若 Def 是 Header 的 PHI（归纳变量），取 Preheader 路径的初值
        Value *Init = V;
        if (PHINode *HeaderPHI = dyn_cast<PHINode>(Def))
          if (HeaderPHI->getParent() == Header)
            Init = HeaderPHI->getIncomingValueForBlock(OutsidePred);

        U.set(Init);
      }
    }

    // ------------------------------------------------------------------
    // 步骤 B：重定向控制流，跳过整个循环
    // ------------------------------------------------------------------
    // 把所有"循环外 -> Header"的边改写为"循环外 -> Exit"。
    // trip count = 0 已证明循环体不执行，可安全直接跳到 Exit。
    //
    // CFG 变化（对照 remove_zero_trip_loop.ll）：
    //   改写前: %entry -> %loop -> %body -> %loop
    //                         \──────────> %exit
    //   改写后: %entry ──────────────────> %exit
    //           （%loop 和 %body 变为不可达，由步骤 C 删除）
    //
    for (BasicBlock *Pred : OutsidePreds) {
      Instruction *TI = Pred->getTerminator();
      for (unsigned i = 0; i < TI->getNumSuccessors(); ++i) {
        if (TI->getSuccessor(i) == Header) {
          TI->setSuccessor(i, Exit);
          break;
        }
      }
    }

    // ------------------------------------------------------------------
    // 步骤 C-1：修复 Exit 的 PHI 节点
    // ------------------------------------------------------------------
    // Header 已不再是 Exit 的前驱，必须：
    //   1) 移除 PHI 中来自 Header 的 incoming
    //   2) 为每个 OutsidePred 添加新的 incoming，保持 SSA 合法
    //
    // 示例（对照 remove_zero_trip_loop.ll）：
    //   改写前: %exit: 隐式地从 %loop 接收 %i
    //   改写后: %exit: 直接从 %entry 接收常量 0（即 %i 的 Preheader 初值）
    //
    for (PHINode &PHI : Exit->phis()) {
      Value *V = PHI.getIncomingValueForBlock(Header);
      // 若 incoming 值是 Header 的 PHI，取其 Preheader 路径的初值
      if (PHINode *HeaderPHI = dyn_cast<PHINode>(V))
        if (HeaderPHI->getParent() == Header)
          V = HeaderPHI->getIncomingValueForBlock(Preheader);
      PHI.removeIncomingValue(Header);
      for (BasicBlock *Pred : OutsidePreds)
        PHI.addIncoming(V, Pred);
    }

    // ------------------------------------------------------------------
    // 步骤 C-2：删除不可达的循环块
    // ------------------------------------------------------------------
    // DTU.flush() 将所有延迟的支配树更新一次性应用。
    // removeUnreachableBlocks() 删除从 Entry 不可达的块（即原循环的
    // Header/Body/Latch）。
    DTU.flush();
    removeUnreachableBlocks(F, &DTU);
    Changed = true;
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // CFG 已修改，但支配树在整个过程中通过 DomTreeUpdater 保持同步更新，
  // 因此可以标记为 preserved，避免后续 pass 重复计算。
  PreservedAnalyses PA;
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
