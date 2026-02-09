# 📘 LLVM Middle-End Optimization Cookbook

**目标读者**：LLVM Pass 开发者  
**核心哲学**：不背诵 API，建立“IR 变换”的直觉。

---

## 目录

| Part | 内容 |
| --- | --- |
| 1 | 核心世界观（IR 三层结构、三大铁律） |
| 2 | 开发者工具箱（指令速查、类型转换、PatternMatch、Utils） |
| 3 | Pass 骨架（New PM） |
| 4 | 局部优化 Cookbook（Peephole） |
| 5 | CFG 优化 Cookbook |
| 6 | PHI 节点与 SSA 维护 |
| 7 | 循环优化（LoopInfo、SCEV） |
| 8 | SROA 与内存优化 |
| 9 | **新手死亡清单**（完整示例与解决方案） |
| 10 | 进阶：官方源码阅读（示例与注释） |

---

## Part 1 · 核心世界观 (Mental Model)

### 1.1 IR 的三层结构

编写 Pass 时，首先确认你在哪一层工作：

* **Module**: 全局变量、函数定义、元数据。
* **Function**: 基本块的集合，参数列表。
* **BasicBlock (BB)**: 控制流节点，指令的线性序列。
* **Instruction (Inst)**: SSA 值，最小操作单位。

### 1.2 必须遵守的三大铁律

1. **SSA 规则**：寄存器（`%1`）只定义一次，不可变。想变？生成新指令。内存（`alloca`）可变，但不是 SSA。
2. **CFG 完整性**：改动跳转（删边/合并块）必须同步更新后续块的 **PHI 节点**。
3. **Def-Use 链**：指令被删除前，必须先处理掉所有使用它的地方（`RAUW`）。

---

## Part 2 · 开发者工具箱 (The Toolkit)

### 2.1 核心指令操作速查表

所有指令继承自 `Instruction` -> `User` -> `Value`。

| 指令类型 | C++ 类名 | 关键判断与访问 API | 典型用途 |
| --- | --- | --- | --- |
| **二元运算** | `BinaryOperator` | `getOpcode()` (e.g. `Instruction::Add`)、`getOperand(0/1)` | 代数化简、常量折叠 |
| **比较** | `ICmpInst` / `FCmpInst` | `getPredicate()` (e.g. `ICMP_EQ`)、`isSigned()`、`isRelational()` | 分支优化、循环边界检查 |
| **分支跳转** | `BranchInst` | `isUnconditional()`、`getCondition()`、`getSuccessor(i)` | CFG 分析、基本块合并 |
| **内存分配** | `AllocaInst` | `getAllocatedType()`、`isArrayAllocation()` | 栈变量分析、Mem2Reg 目标 |
| **内存读写** | `LoadInst` / `StoreInst` | `getPointerOperand()`、`getAlign()`、`isVolatile()` | 别名分析、死存储消除 (DSE) |
| **地址计算** | `GetElementPtrInst` | `getSourceElementType()`、`hasAllConstantIndices()`、`idx_begin/end()` | 结构体/数组访问、SROA 核心 |
| **汇聚值** | `PHINode` | `getIncomingValue(i)`、`getIncomingBlock(i)`、`addIncoming(Val, BB)` | 处理控制流汇合 |
| **类型转换** | `CastInst` | `getSrcTy()`、`getDestTy()`；子类 `BitCast`、`ZExt`、`Trunc` | 类型双关、位宽调整 |
| **函数调用** | `CallInst` | `getCalledFunction()`、`arg_begin/end()` | 内联 (Inlining)、过程间分析 |

### 2.2 类型转换 (Casting)

LLVM 禁止使用 C++ `dynamic_cast`，请使用以下高性能模板：

* **`isa<T>(Val)`**: 检查类型。 `if (isa<BinaryOperator>(I)) ...`
* **`cast<T>(Val)`**: 强制转换（失败会 Assertion Crash）。用于你 100% 确定的场景。
* **`dyn_cast<T>(Val)`**: 尝试转换，失败返回 `nullptr`。**最常用**。

### 2.3 模式匹配 (PatternMatch)

神器！包含在 `#include "llvm/IR/PatternMatch.h"`。不要手写 `if-else` 检查 Opcode。

```cpp
using namespace llvm::PatternMatch;
Value *X, *Y;

// 匹配: X + 0
if (match(I, m_Add(m_Value(X), m_Zero()))) { ... }

// 匹配: (X * Y) + X  (乘法分配律逆运算)
if (match(I, m_Add(m_Mul(m_Value(X), m_Value(Y)), m_Deferred(X)))) { ... }

// 匹配: 具体的常量 42
if (match(I, m_SpecificInt(42))) { ... }
```

### 2.4 必备 Utils 函数库

不要造轮子，LLVM 自带了大量工具函数。

**文件**: `llvm/Transforms/Utils/Local.h` & `BasicBlockUtils.h`

* **`RecursivelyDeleteTriviallyDeadInstructions(I)`**: 删掉指令 I，如果它的操作数也没人用了，顺便把操作数也删掉（递归清理）。
* **`SimplifyInstruction(I, DL)`**: 尝试计算指令结果，但不修改 IR。例如输入 `add i32 1, 2` 返回 `ConstantInt(3)`。
* **`MergeBlockIntoPredecessor(BB)`**: 将 BB 合并入前驱。
* **`SplitEdge(BB1, BB2)`**: 在两个块中间插入一个新块（用于安插代码）。
* **`ReplaceInstWithInst(Old, New)`**: 替换并维护 Def-Use 链。

---

## Part 3 · Pass 骨架 (New PM)

```cpp
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PatternMatch.h"

using namespace llvm;

struct MyPass : PassInfoMixin<MyPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    bool Changed = false;

    // 1. 获取分析结果 (如果需要)
    // auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);

    // 2. 遍历 BasicBlocks
    for (BasicBlock &BB : F) {
        // 3. 遍历 Instructions (使用 make_early_inc_range 防止迭代器失效)
        for (Instruction &I : make_early_inc_range(BB)) {
            // 业务逻辑...
            if (optimize(I)) Changed = true;
        }
    }

    // 4. 返回结果状态
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
```

---

## Part 4 · 局部优化 Cookbook (Peephole)

**场景**：`InstCombine` 风格的代数化简。
**核心动作**：匹配 -> 创建新值 -> RAUW -> 删除旧值。

**示例：将 `sub X, 0` 优化为 `X`**

```cpp
bool tryOptimize(Instruction &I) {
    using namespace llvm::PatternMatch;
    Value *X;
    
    // 1. 匹配
    if (match(&I, m_Sub(m_Value(X), m_Zero()))) {
        // 2. 替换 (RAUW)
        I.replaceAllUsesWith(X);
        // 3. 标记删除 (不要立即 erase，放入 worklist 或由调用者处理)
        I.eraseFromParent(); 
        return true;
    }
    return false;
}

```

---

## Part 5 · CFG 优化 Cookbook

**场景**：清理多余的跳转，合并基本块。
**核心动作**：修改 Branch 目标，维护 PHI，合并 Block。

### 5.1 遍历图关系

```cpp
// 谁跳到了我？
for (BasicBlock *Pred : predecessors(&BB)) { ... }
// 我跳到了谁？
for (BasicBlock *Succ : successors(&BB)) { ... }
```

### 5.2 基本块合并 (Merging)

**条件**：`BB_A` 无条件跳转到 `BB_B`，且 `BB_B` 只有 `BB_A` 一个前驱。

```cpp
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

bool tryMergeBlock(BasicBlock *BB) {
    BasicBlock *Pred = BB->getSinglePredecessor();
    if (!Pred) return false;
    
    // 确保前驱是无条件跳转
    BranchInst *BI = dyn_cast<BranchInst>(Pred->getTerminator());
    if (!BI || !BI->isUnconditional()) return false;
    
    // 确保 Pred 只有这一个后继 (虽然 unconditional 隐含了这点，但检查更安全)
    if (BI->getSuccessor(0) != BB) return false;

    // 一键合并
    MergeBlockIntoPredecessor(BB);
    return true;
}

```

---

## Part 6 · PHI 节点与 SSA 维护

**场景**：修改 CFG 后，必须修复 PHI 节点。

### 6.1 删除前驱时的 PHI 修复

如果你删除了 `BB_A -> BB_B` 的边：

```cpp
// 在 BB_B 中
BasicBlock *BB_B = ...;
BasicBlock *BB_A = ...; // 被移除的前驱

for (PHINode &PN : BB_B->phis()) {
    // 告诉 PHI：不会再有控制流从 BB_A 过来了
    PN.removeIncomingValue(BB_A);
}

```

### 6.2 插入新值

如果你给 BB 增加了一个新的前驱 `NewPred`：

```cpp
PHINode *PN = ...;
// 必须告诉 PHI：如果从 NewPred 过来，值应该是多少？
// 如果是 LCSSA 场景，可能需要 Undef 或者对应的值
PN->addIncoming(SomeValue, NewPred);

```

---

## Part 7 · 循环优化 (Loop Optimization)

**场景**：循环不变量外提 (LICM)、循环展开 (Unroll)。
**依赖分析**：`LoopInfo`, `ScalarEvolution` (SCEV), `DominatorTree`.

### 7.1 获取循环信息

```cpp
// 在 Pass 中获取 Analysis
LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

for (Loop *L : LI) {
    // 获取关键块
    BasicBlock *Header = L->getHeader();
    BasicBlock *Preheader = L->getLoopPreheader(); // 可能为 null
    BasicBlock *Latch = L->getLoopLatch();
    
    // 遍历子循环
    for (Loop *SubLoop : *L) { ... }
}

```

### 7.2 Scalar Evolution (SCEV)

SCEV 是 LLVM 的数学引擎，能推导循环变量的公式。

* **计算循环次数 (Trip Count)**:
```cpp
ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
const SCEV *TripCount = SE.getBackedgeTakenCount(L);
// 检查是否是常数
if (auto *C = dyn_cast<SCEVConstant>(TripCount)) {
    uint64_t Iterations = C->getValue()->getZExtValue();
}
```


* **判断是否是归纳变量**:
`SE.isLoopInvariant(Val, L)`

---

## Part 8 · SROA 与 内存优化

**场景**：将 `alloca` (栈内存) 提升为寄存器 (SSA)。
**核心逻辑**：SROA = Mem2Reg + 结构体拆解。

### 8.1 开发者需要做的

通常你不需要自己写 SROA，但你需要生成**SROA 友好**的代码：

1. **使用标准类型**：尽量避免疯狂的指针强转（虽然 SROA 能处理 `bitcast`，但越简单越好）。
2. **避免指针逃逸**：如果把 `alloca` 的地址传给了外部函数（`call @foo(ptr %ptr)`），SROA 就无法提升它，因为外部函数可能修改内存。

---

## Part 9 · 新手死亡清单 (Death Checklist) ☠️

在提交代码前，请逐项检查。每项均附**错误示例**与**正确写法/解决方案**。

---

### 9.1 迭代器失效 (Iterator Invalidation)

**问题**：在遍历 `BasicBlock` 时删除指令，会使迭代器失效，导致未定义行为或崩溃。

**❌ 错误示例**

```cpp
// 遍历 BB 时直接 erase，++it 会访问已失效的迭代器
for (Instruction &I : BB) {
  if (isDead(&I)) {
    I.eraseFromParent();  // 迭代器 I 失效！
  }
}
// 可能表现：崩溃、漏删、重复处理
```

**✅ 正确写法**

```cpp
#include "llvm/IR/Instructions.h"

// 使用 make_early_inc_range：在 ++ 之前先推进迭代器，再处理当前元素
for (Instruction &I : make_early_inc_range(BB)) {
  if (isDead(&I)) {
    I.eraseFromParent();  // 安全：迭代器已“提前”指向下一个
  }
}
```

**替代方案**：先收集要删除的指令，再统一删除。

```cpp
SmallVector<Instruction *, 8> ToRemove;
for (Instruction &I : BB)
  if (isDead(&I))
    ToRemove.push_back(&I);
for (Instruction *I : ToRemove)
  I->eraseFromParent();
```

---

### 9.2 PHI 节点孤儿 (Orphaned PHI Incoming)

**问题**：修改了控制流（删边、改 terminator、合并块）后，目标块里 PHI 仍引用已删除的前驱块，导致非法 IR 或验证失败。

**❌ 错误示例**

```cpp
// 想把 BB_A 的 terminator 从 "br BB_B" 改成 "br BB_C"，直接替换
BranchInst *BI = cast<BranchInst>(BB_A->getTerminator());
BasicBlock *OldSucc = BI->getSuccessor(0);  // BB_B
// 直接改后继，没有更新 BB_B 的 PHI
BI->setSuccessor(0, BB_C);
// BB_B 的 PHI 里还有 addIncoming(..., BB_A)，但 BB_A 不再跳到 BB_B → 非法
```

**✅ 正确做法：先更新 PHI，再改跳转**

```cpp
BasicBlock *BB_A = ...;
BasicBlock *BB_B = BI->getSuccessor(0);
BasicBlock *BB_C = ...;

// 1. 在 BB_B 中移除来自 BB_A 的 PHI 入边
for (PHINode &PN : BB_B->phis())
  PN.removeIncomingValue(BB_A);

// 2. 若 BB_C 有 PHI，需要为来自 BB_A 的边添加正确入边（此处示例为 Undef）
for (PHINode &PN : BB_C->phis())
  PN.addIncoming(UndefValue::get(PN.getType()), BB_A);

// 3. 再修改 terminator
BI->setSuccessor(0, BB_C);
```

**合并块时的 PHI**：使用 `MergeBlockIntoPredecessor` 等工具函数会自动处理 PHI；手写合并时务必把被合并块中 PHI 的 incoming 迁移到新位置。

---

### 9.3 错误的类型替换 (Type Mismatch in RAUW)

**问题**：`replaceAllUsesWith(NewVal)` 要求 `NewVal` 与当前 `Value` 的 **类型完全一致**（包括 `i32` vs `i64`、`ptr` 层级）。类型不一致会触发断言或生成非法 IR。

**❌ 错误示例**

```cpp
// I 是 i32，想用常量 42 替换，但 42 常量的类型写成了 i64
Value *I = ...;  // 类型为 i32
Value *NewVal = ConstantInt::get(Type::getInt64Ty(Ctx), 42);
I->replaceAllUsesWith(NewVal);  // 断言失败或非法 IR：i32 的 use 不能接 i64
```

**✅ 正确写法**

```cpp
// 新值的类型必须与旧值一致
Type *Ty = I->getType();
Value *NewVal = ConstantInt::get(Ty, 42);
I->replaceAllUsesWith(NewVal);
```

**需要类型不同时**：先插入类型转换指令，再用转换结果做 RAUW。

```cpp
IRBuilder<> Builder(&I);
Value *Cast = Builder.CreateIntCast(NewVal, I->getType(), /*isSigned*/ true);
I->replaceAllUsesWith(Cast);
I->eraseFromParent();
```

---

### 9.4 忽视 Unreachable 块 (Unreachable Blocks)

**问题**：并非所有 BasicBlock 都能从 Entry 到达。在不可达块上做“图遍历”“前驱/后继”假设可能死循环、除零或访问非法内存。

**❌ 错误示例**

```cpp
// 假设“从 Entry 做 BFS 能到达所有块”
BasicBlock *Entry = &F.getEntryBlock();
SmallPtrSet<BasicBlock *, 16> Visited;
std::queue<BasicBlock *> Q;
Q.push(Entry);
while (!Q.empty()) {
  BasicBlock *BB = Q.front();
  Q.pop();
  if (!Visited.insert(BB).second) continue;
  for (BasicBlock *Succ : successors(BB))
    Q.push(Succ);  // 若有不可达块形成自环，不会入队，但...
}
// 之后对所有 BB 做变换，若假设“所有 BB 都在 Visited”会出错
for (BasicBlock &BB : F) {
  transform(&BB);  // 不可达块可能未被正确初始化/处理
}
```

**✅ 正确做法**

- 使用 **DominatorTree**：支配树只包含从 Entry 可达的块；在 DT 上迭代即隐含“仅可达块”。
- 需要处理不可达块时，用 **`removeUnreachableBlocks(F)`**（`llvm/Transforms/Utils/Local.h`）先清理，再跑自己的分析。

```cpp
#include "llvm/Transforms/Utils/Local.h"

// 若 Pass 不关心不可达块，先删掉再处理
removeUnreachableBlocks(F);

// 或：只对支配树中的块做变换
DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
for (BasicBlock *BB : depth_first(DT.getRootNode()))
  transform(BB);
```

---

### 9.5 滥用 replaceAllUsesWith (RAUW 误用)

**问题**：对自身 RAUW、或对尚未插入到 Function 的指令做 RAUW，会导致非法状态或崩溃。

**❌ 错误示例 1：对自己替换自己**

```cpp
Instruction *I = ...;
I->replaceAllUsesWith(I);  // 无意义，且某些路径下可能出问题
```

**❌ 错误示例 2：新指令未插入就 RAUW**

```cpp
Instruction *Old = ...;
Instruction *New = BinaryOperator::CreateAdd(X, Y);  // 未插入到任何 BB
Old->replaceAllUsesWith(New);  // New 无 parent，后续 Pass 可能崩溃
Old->eraseFromParent();       // 若 New 仍无 parent，IR 非法
```

**✅ 正确流程**

```cpp
// 1. 创建新指令并插入到合适位置（如 Old 之前）
IRBuilder<> Builder(Old);
Value *New = Builder.CreateAdd(X, Y);

// 2. 再替换并删除旧指令
Old->replaceAllUsesWith(New);
Old->eraseFromParent();
```

**小结**：RAUW 前确保 (1) 新值已插入到 IR 中；(2) 新值与旧值类型一致；(3) 不要用同一 Value 替换自己（除非文档明确允许）。

---

## Part 10 · 进阶：如何阅读官方源码

不要只看文档，官方 Pass 是最好的老师。下面给出**推荐文件路径**、**示例代码片段**与**阅读注释**，便于按图索骥。

---

### 10.1 InstCombine：PatternMatch 与代数化简

**路径**：`llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp`（以及同目录下 `InstCombineMulDivRem.cpp`、`InstCombineSimplifyDemanded.cpp` 等）。

**示例：`X + 0 => X` 的官方实现思路**

```cpp
// 文件: InstCombineAddSub.cpp
// 函数: InstCombiner::visitAdd() 中的一段

// 使用 PatternMatch 匹配 "add X, 0" 或 "add 0, X"
Value *X;
if (match(Op1, m_Zero()) || match(Op0, m_Zero())) {
  // Op0/Op1 中非零的那个即为 X
  Value *NonZero = match(Op1, m_Zero()) ? Op0 : Op1;
  return replaceInstUsesWith(I, NonZero);  // RAUW + 由调用者统一删 I
}
```

**阅读要点**：

- 所有化简都通过 `replaceInstUsesWith(I, NewVal)` 替换，旧指令由 InstCombine 的 worklist 统一删除，避免迭代器失效。
- 复杂模式（如 `(X * Y) + X => X * (Y + 1)`）见同文件或 `InstCombineMulDivRem.cpp`，重点看 `m_Deferred`、`m_One()` 等组合用法。

---

### 10.2 SimplifyCFG：安全改 CFG 与 PHI

**路径**：`llvm/lib/Transforms/Utils/SimplifyCFG.cpp`。

**示例：合并仅有一个前驱且前驱仅有一个后继的块**

```cpp
// 文件: SimplifyCFG.cpp
// 函数: simplifyCFG() 中关于 MergeBlockIntoPredecessor 的用法

// 条件：BB 有唯一前驱 Pred，且 Pred 唯一后继就是 BB
if (BasicBlock *Pred = BB->getSinglePredecessor()) {
  if (Pred != BB && (BranchInst *)Pred->getTerminator()->getSuccessor(0) == BB) {
    MergeBlockIntoPredecessor(BB);  // 内部会处理 PHI：把 BB 的 PHI 合并到 Pred
    return true;
  }
}
```

**示例：删除不可达块**

```cpp
// 同文件内会调用 removeUnreachableBlocks()
// 定义在: llvm/lib/Transforms/Utils/Local.cpp
// 作用：从 Entry 做 DFS，删掉所有不可达的 BB，并正确更新 PHI 的 incoming
removeUnreachableBlocks(F);
```

**阅读要点**：

- 任何“改 terminator / 删边 / 合并块”后，注意后续如何更新 PHI；SimplifyCFG 里大量 `removeIncomingValue` / `addIncoming` 的成对使用。
- Switch 转跳表、HoistCommonCode 等在同一文件，可顺带学习“在 CFG 上安全插块”的写法（如 `SplitBlockPredecessors`）。

---

### 10.3 SROA：内存切片与指针重写

**路径**：`llvm/lib/Transforms/Scalar/SROA.cpp`。

**示例：判断 alloca 是否可被 SROA 提升（简化逻辑）**

```cpp
// 文件: SROA.cpp
// 思路：若 alloca 的地址被 store 到全局、或传给未知函数，则“逃逸”，不能整块提升

bool isAllocaPromotable(AllocaInst *AI) {
  for (Use &U : AI->uses()) {
    if (StoreInst *SI = dyn_cast<StoreInst>(U.getUser())) {
      if (isa<GlobalVariable>(SI->getPointerOperand()))
        return false;  // 存到全局，逃逸
    }
    if (CallInst *CI = dyn_cast<CallInst>(U.getUser())) {
      if (!CI->getCalledFunction() || CI->getCalledFunction()->isDeclaration())
        return false;  // 传给未知函数，逃逸
    }
  }
  return true;
}
```

**阅读要点**：

- 实际 SROA 会按“切片”（slice）拆分 alloca，再为每个切片生成新的 alloca 或 SSA 值，并重写所有 Load/Store/GEP。重点看如何用 `Slice` 结构和 `partitionUse` 一类函数划分用途。
- 难度高，建议先掌握 Part 6（PHI）和 Part 8（SROA 友好 IR），再回头细读。

---

### 10.4 ScalarEvolution：归纳与 Trip Count

**路径**：`llvm/lib/Analysis/ScalarEvolution.cpp`，以及使用处如 `llvm/lib/Transforms/Scalar/LoopUnrollPass.cpp`。

**示例：在 Pass 里用 SCEV 求循环迭代次数**

```cpp
// 在 LoopPass 或 FunctionPass 中
ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
Loop *L = ...;

const SCEV *BackedgeCount = SE.getBackedgeTakenCount(L);
// 若为常数，可得到迭代次数
if (auto *C = dyn_cast<SCEVConstant>(BackedgeCount)) {
  uint64_t TripCount = C->getValue()->getZExtValue();
  // 实际迭代次数 = TripCount + 1（取决于 getBackedgeTakenCount 的语义）
}
```

**官方源码中的典型用法（带注释）**

```cpp
// 文件: LoopUnrollPass.cpp 或 LoopVectorize.cpp
// 获取循环回边次数
const SCEV *BECount = SE.getBackedgeTakenCount(L);
if (isa<SCEVCouldNotCompute>(BECount))
  return;  // 无法计算，放弃优化

// 判断某值在循环内是否不随迭代变化
if (SE.isLoopInvariant(SE.getSCEV(Val), L))
  // 可做 LICM 等
```

**阅读要点**：

- `getSCEV(Value *)` 把 IR 值映射到 SCEV 表达式；`getBackedgeTakenCount` 是“回边执行次数”的数学形式。
- `ScalarEvolution.cpp` 体量大，可先看 `getSCEV`、`getBackedgeTakenCount`、`isLoopInvariant` 的入口与文档注释，再根据需要深入递归求值部分。

---

### 10.5 推荐阅读顺序小结

| 顺序 | 文件/目录 | 目的 |
| --- | --- | --- |
| 1 | `InstCombine/InstCombineAddSub.cpp` | PatternMatch + RAUW 规范写法 |
| 2 | `Utils/SimplifyCFG.cpp` | CFG 修改与 PHI 维护、Unreachable 清理 |
| 3 | `Utils/Local.cpp` | `removeUnreachableBlocks`、`MergeBlockIntoPredecessor` 等工具实现 |
| 4 | `Scalar/SROA.cpp` | 内存分析、切片与重写（进阶） |
| 5 | `Analysis/ScalarEvolution.cpp` + `LoopUnrollPass.cpp` | SCEV 使用方式与循环分析 |
