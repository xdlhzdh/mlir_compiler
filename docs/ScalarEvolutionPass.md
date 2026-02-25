# ScalarEvolutionAnalysis 是什么

在 LLVM 的新 PassManager 里：

- **ScalarEvolutionAnalysis** 是一个 **Analysis Pass**，不是 Transform Pass
- 它专门分析某个 Function **内循环里的整数标量**
- 本质上返回的是一个 `ScalarEvolution` 对象引用，存储了整个函数的循环数学模型

因此：

```cpp
auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
```

返回的 `SE` 本质上是 `F` 对应的 `ScalarEvolution` 实例，它已经：

- 分析了 F 中所有循环的 phi 节点
- 对每个 loop 生成递推式（SCEV Expr）
- 提供方法获取 backedge taken count / trip count / recurrence

---

# SE 能做什么？方法举例

```cpp
LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

for (Loop *L : LI) {
    const SCEV *BTC = SE.getBackedgeTakenCount(L);
    const SCEV *Start = SE.getSCEV(L->getHeader()->getFirstNonPHI());
}
```

**解释：**

| 方法 | 含义 |
|------|------|
| `getBackedgeTakenCount(L)` | 返回循环迭代次数 - 1 |
| `getSCEV(Value*)` | 返回该值在循环中的 SCEV 表达式（递推式） |

---

# 具体例子

## 源码示例

```c
int sum_to_n(int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += i;
    }
    return sum;
}
```

## LLVM IR（简化）

```llvm
entry:
  %sum = alloca i32
  store i32 0, i32* %sum
  %i = alloca i32
  store i32 0, i32* %i
  br label %loop

loop:
  %iv = load i32, i32* %i
  %cmp = icmp slt i32 %iv, %n
  br i1 %cmp, label %body, label %exit

body:
  %sum_val = load i32, i32* %sum
  %new_sum = add i32 %sum_val, %iv
  store i32 %new_sum, i32* %sum
  %i_next = add i32 %iv, 1
  store i32 %i_next, i32* %i
  br label %loop

exit:
  %ret = load i32, i32* %sum
  ret i32 %ret
```

## 调用 SE 分析

```cpp
auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
auto &LI = AM.getResult<LoopAnalysis>(F);

for (Loop *L : LI) {
    const SCEV *BTC = SE.getBackedgeTakenCount(L);
    BTC->print(errs());  // 打印 SCEV 表达式
}
```

## 结果解释

- **BTC** 表示循环迭代次数 - 1
- 输出大概类似：`{0,+,1}<loop>`

**含义：**

- `{0,+,1}` → 从 0 开始，每次 +1
- `<loop>` → 这个表达式属于哪个 Loop
- **Backedge taken count** = n - 1

因此可以得出：

- 循环执行次数 = n
- i 的递推式 = `{0,+,1}<loop>`
- sum 的值依赖于 i → 可以进行 LICM / GVN / vectorization

---

# 总结

`AM.getResult<ScalarEvolutionAnalysis>(F)` 返回函数 F 的 `ScalarEvolution` 对象。

**SE 中保存了：**

- 所有循环的 induction variable
- 循环 trip count（backedge taken count）
- 每个值在循环中的 SCEV 表达式

**用法：**

```cpp
const SCEV *BTC = SE.getBackedgeTakenCount(L);
const SCEV *Expr = SE.getSCEV(Value* v);
```

**SCEV 可以配合：**

- **LoopInfo** → 找到循环
- **DominatorTree** → 安全修改 CFG
- **MemorySSA** → 安全修改 load/store

---

# 核心理解

SE 就是 Function F **内循环数学模型的缓存 + 分析结果**。它不是每次访问时重新计算，而是把所有循环变量的递推式提前生成好，供 Pass 使用。
