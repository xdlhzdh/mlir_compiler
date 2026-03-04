# MLIR L1 Graph Level 学习笔记：StableHLO IR 语义理解

> 本笔记聚焦 **IR 语义理解能力**（不涉及写 Pass），面向 StableHLO/MHLO 的 op 语义、shape/rank、broadcast、dynamic shape 以及 side effect 等核心概念。

---

## 一、StableHLO 概览

StableHLO 是一套**高层算子集（High-Level Operations）**，作为 ML 框架（TensorFlow、JAX、PyTorch）与 ML 编译器（XLA、IREE）之间的**可移植层**。语义由 [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) 定义，是 verifier 与 shape 推导的权威依据。

- **MHLO**：TensorFlow/MLIR 历史命名，与 StableHLO 同源，语义基本一致。
- 理解 IR 时以 **StableHLO 规范**为准，本文中的 op 以 `stablehlo.` 前缀为例。

---

## 二、Shape 与 Rank 语义

### 2.1 基本定义

- **Tensor 类型**：`tensor<shape element_type>`，例如 `tensor<2x3xf32>`。
- **Shape**：各维度的**非负整数大小**，按维度从 0 到 R-1 排列。
- **Rank（秩）**：维度个数 R。例如 `tensor<2x3xf32>` 的 rank = 2，shape = `[2, 3]`。
- **Axis / Dimension**：常互换使用，指某一维的编号（0 到 R-1）。

标量在 StableHLO 中用 **0 维 tensor** 表示，例如 `tensor<f32>`，rank = 0。

### 2.2 常见约束

- **SameOperandsAndResultShape**：如 `add`、`multiply`，要求操作数与结果 **shape 完全一致**。
- **SameType**：操作数与结果 **类型完全一致**（shape + element type）。
- 很多 op 的**结果 shape** 可由操作数 shape 与属性**唯一推导**；少数 op（如部分 dynamic 版本）结果类型无法仅从操作数推导，需由 IR 显式写出。

### 2.3 约束是什么、怎么用

- **约束（constraints）**：规范对某个 op 规定的**操作数/结果在类型、shape 等方面必须满足的条件**。用于合法性验证和类型/shape 推导。
- **读图 / 写 IR**：不需要“调用”约束，而是**遵守**约束。写 IR 时保证每个 op 满足其约束（如 `add` 要求操作数与结果 shape 一致），否则 verifier 会报错；例如“标量+张量”要先 `broadcast_in_dim` 再 `add`。
- **定义新 op（ODS）**：在 TableGen 里用 **Trait**（如 `SameOperandsAndResultShape`、`SameType` 或 `PredOpTrait`）声明约束，MLIR 会据此生成 verifier。
- **写 Pass**：可假定已有 IR 已满足约束；做变换时**你创建或替换出的新 op**，必须满足**该算子类型在规范中的约束**（例如新创建的 `add` 仍须操作数与结果 shape 一致），否则 verifier 会报错。

---

## 三、Broadcast 规则

### 3.1 元素级二元 op 的隐式 broadcast

对于 `add`、`multiply`、`divide` 等元素级二元 op，规范要求 **lhs、rhs、result 具有相同 type**，即 **同一 shape**。因此若框架要表达“标量与张量相加”等，会先通过 **broadcast** 把标量/小 shape 张量扩成与另一操作数相同的 shape，再做元素级运算。

### 3.2 broadcast_in_dim（显式 broadcast）

**说人话**：这个 op 做一件事：把**输入张量（operand）**按你给的规则**拉成**你想要的**输出张量（result）**。你要提供两样东西，彼此独立：

1. **结果的 shape**：即**操作完成后输出的张量**的 shape，是你在 IR 里**显式写出的**（或由类型推断得到），不是由 broadcast_dimensions 算出来的。例如写 `-> tensor<2x3xf32>` 就表示输出是 2×3。
2. **broadcast_dimensions**：一个整数列表，只负责说明「**输入的每一维，对应到输出的哪一维**」，不决定输出是几维、每一维多长。

- **Shape 和“维”**：shape 表示各维长度，如 `[2, 3]` = 第 0 维长 2、第 1 维长 3（可理解为 2 行 3 列）。维编号从 0 开始。
- **broadcast_dimensions 只做“维的对应”**：列表第 k 个数字 = 输入的第 k 维 对应 输出的第几维。
  - `[0, 1]`：输入第 0 维→输出第 0 维，输入第 1 维→输出第 1 维。
  - `[2, 1]`：输入第 0 维→输出第 2 维，输入第 1 维→输出第 1 维（所以输出至少是 3 维，因为用到了第 2 维）。
- **复制规则**：若输入在某维长度是 1，而输出在**对应到的那一维**长度大于 1，就在那一维上**复制**填满；若输入某维不是 1，则输出在对应维上**长度必须和输入相同**，逐元素对应，不复制。

**举例（把因果说清楚）**：

- 输入 shape `[1, 3]`（1 行 3 列，一行 `[a,b,c]`）。
- **我们想要**输出是 2 行 3 列，所以**在 IR 里把结果的 shape 写成 `[2, 3]`**（这是“操作后的结果”的 shape，不是“维度扩张”的中间概念）。
- 再指定 `broadcast_dimensions = [0, 1]`，表示：输入第 0 维对输出第 0 维，输入第 1 维对输出第 1 维。
- 输入第 0 维（行）长 1，输出第 0 维（行）长 2 → 在行方向复制一份，得到两行都是 `[a,b,c]`。输入第 1 维（列）长 3，输出第 1 维也长 3 → 直接对齐，不复制。  
  所以：**结果 shape 是“操作后的结果”的 shape；`[2,3]` 是我们事先给定的目标，不是由 `[0,1]` 推出来的。**

**再举一例：`broadcast_dimensions = [2, 1]`**

- 输入仍是 shape `[1, 3]`（一行 `[a, b, c]`）。
- 这次**我们想要**输出是 3 维的，例如 **结果 shape 写成 `[2, 3, 2]`**（可以想成 2 个 3×2 的“片”）。
- 指定 `broadcast_dimensions = [2, 1]`：输入第 0 维 → 结果第 2 维，输入第 1 维 → 结果第 1 维。也就是说，**输入的“行”对到结果的第 2 维，“列”对到结果的第 1 维**；结果的第 0 维没有输入维对应。
- 对应关系与复制：
  - 结果第 1 维长 3，和输入第 1 维（列）对齐，不复制。
  - 结果第 2 维长 2，输入第 0 维（行）长 1 → 在这一维上复制：同一行 `[a,b,c]` 沿第 2 维复制成 2 份，得到一块 3×2（3 行 2 列，其实每列都是 [a,b,c]）。
  - 结果第 0 维长 2，没有输入维对应 → 整块 3×2 沿第 0 维再复制 2 份，得到 2×3×2。  
    所以 `[2, 1]` 的效果是：**把输入的那一行“塞进”结果的第 1、2 维（列对列，行在结果第 2 维上从 1 复制成 2），再沿第 0 维整块复制。**

**形式化**（供查阅）：

```text
result[i0, i1, ..., iR-1] = operand[j0, j1, ..., jR'-1]
```

其中对 operand 的每个维度 k：`jk = (dim(operand, k) == 1) ? 0 : i[broadcast_dimensions[k]]`。即：operand 某维为 1 则取 0（沿该轴复制）；否则取结果在 `broadcast_dimensions[k]` 维上的下标。这里dim指的是dimension(维度)。

**约束要点**：

- `size(broadcast_dimensions) = rank(operand)`。
- `broadcast_dimensions` 中各维互不相同，且取值在 `[0, rank(result))`。
- 对每个 operand 维度 j：要么 `dim(operand, j) == 1`，要么 `dim(operand, j) == dim(result, broadcast_dimensions[j])`。
- 元素类型一致（且不允许 complex 与 non-complex 互转）。

**示例**：

```mlir
// operand: [[1, 2, 3]]  shape (1, 3)
// 映射到 result 的 dim 2 和 dim 1；dim 0 为扩展维
%result = stablehlo.broadcast_in_dim %operand {
  broadcast_dimensions = dense<[2, 1]> : tensor<2xi64>
} : (tensor<1x3xi32>) -> tensor<2x3x2xi32>
// result dim 0 被“复制”成 2；dim 1、2 对应 operand 的 dim 1、0
```

理解要点：**broadcast_dimensions[k]** 表示“operand 的第 k 维对应 result 的哪一维”；若 operand 该维为 1，则在 result 的该维上复制。

### 3.2.1 和 broadcast 的区别：用 0（或常数）填充 —— pad

**broadcast_in_dim 只会“复制”原数据**，不会在多出来的位置填 0。若要**在边缘或空隙里塞进 0（或别的常数）**，用的是另一个 op：**`stablehlo.pad`**。

- **语义**：在每一维的**低端**和**高端**各加若干格，这些新格用你给的 **padding_value**（标量）填满，常见是 0。
- **参数**：`operand`、`padding_value`（标量）、`edge_padding_low`（每维低端 pad 多少）、`edge_padding_high`（每维高端 pad 多少），以及可选的 `interior_padding`（维内插空，不常用）。
- **结果**：原张量不变，四周（或指定边）多出一圈，多出来的元素 = `padding_value`。

**举例（塞 0）**：

- 输入 shape `[2, 2]`，即 2×2 矩阵。
- 想在每一维**低端加 1 格、高端加 1 格**，多出来的地方填 0。
- 则 `edge_padding_low = [1, 1]`，`edge_padding_high = [1, 1]`，`padding_value = 0`（或 constant 0）。
- 结果 shape 为 `[4, 4]`：中间 2×2 是原数据，上下左右各多 1 格，全是 0。

```text
原 [2,2] 数据:     pad 后 (4,4)，padding_value=0：
  a b                 0 0 0 0
  c d       →         0 a b 0
                      0 c d 0
                      0 0 0 0
```

所以：**要“复制扩展”用 broadcast_in_dim，要“边上塞 0（或常数）”用 pad。**

### 3.3 与 reduce、LayerNorm 等的关系

规范中 LayerNorm、BatchNorm 等都用 **reduce + broadcast_in_dim** 描述：先在 feature 维上 reduce 得到 1D 统计量，再用 `broadcast_in_dim(..., [feature_index], shape(operand))` 把统计量扩回 operand 的 shape 做归一化。掌握 `broadcast_in_dim` 的“维映射 + 大小为 1 则复制”即可读懂这类模式。

---

## 四、Dynamic Shape 语义

### 4.1 动态维的表示

- **未知维大小**：在类型中用 `?` 表示，如 `tensor<16x?xf32>` 表示第 1 维在编译时未知。
- **Bounded dynamism**：有已知上界的动态维，可用 `#stablehlo.bounds` 等编码表示上界（用于分配、padding 等）。
- **Unbounded dynamism**：无上界，常见于 dynamic batch、dynamic sequence length。

### 4.2 三类概念

1. **Shape polymorphism（形状多态）**
   - 所有动态性仅与 **tensor 的 shape** 相关，且可追溯到**输入参数**。
   - 一旦输入 shape 确定，理论上可把整图 refine 成静态 shape（通过 `stablehlo-refine-shapes`、`stablehlo-canonicalize-dynamism` 等）。

2. **Data-dependent dynamism**
   - 维大小依赖**数据内容**（如 `nonzeros` 输出长度依赖非零元个数）。
   - 常通过 bounded dynamism + padding 在硬件上实现。

3. **Dynamic ops**
   - 如 `dynamic_slice`、`dynamic_update_slice`、`dynamic_broadcast_in_dim` 等，**起始索引或输出 shape 在运行时由操作数提供**，而不是仅由属性给出。

### 4.3 dynamic_broadcast_in_dim

- 与 `broadcast_in_dim` 语义一致，但 **result 的 shape 由运行时操作数给出**（例如另一操作数表示 `result_shape`）。
- 类型推断：结果类型无法仅从操作数和属性完全推断，需在 IR 中显式写出或由上层推导。
- 典型用法：dynamic batch 时，把标量或小 shape 常量按“当前 batch 的 shape”做 broadcast。

### 4.4 从动态到静态的 Pass 顺序示例

文档中常用顺序：

1. `stablehlo-refine-arguments`：用具体类型替换入口参数（如 `tensor<?xf32>` → `tensor<16xf32>`）。
2. `stablehlo-refine-shapes`：在整图中传播新 shape，更新类型。
3. `stablehlo-canonicalize-dynamism`：在 shape 已完全静态处，将 `dynamic_*` op 替换为静态版本（如 `dynamic_broadcast_in_dim` → `broadcast_in_dim`）。

---

## 五、Side Effect 与 Pure Op

### 5.1 概念

- **Pure op**：无副作用，且可**随意重排、复制、消除**（在语义等价前提下），例如 `add`、`multiply`、`reshape`、`broadcast_in_dim`（在无错误前提下）。
- **Side effect**：会读/写外部状态或与其他进程通信，如 `send`/`recv`、`infeed`/`outfeed`、collective（`all_reduce`、`all_gather` 等）。这些 op 的执行顺序、是否可 speculated 需按规范与接口来理解。

StableHLO 通过 MLIR 的 **SideEffectOpInterface** 等机制标注；多数元素级运算和“形状/类型变换”类 op 视为无副作用，collective 与 IO 视为有副作用。

### 5.2 Speculatability

- **Speculatable**：可被“ speculated”（提前执行、重排）而不改变程序语义。
- 多数 StableHLO op 具有 `HLO_SpeculatableIf*` 一类 trait；允许 dynamic shape 的 op 可能因 shape 不匹配而不可 speculate，会使用 **ConditionallySpeculatable** 并在接口中实现条件判断。
- 理解 IR 时：**纯计算 + 无 token 依赖** 的 op 一般可重排；**token 依赖、collective、IO** 会约束执行顺序。

---

## 六、重点 Op 语义精读

### 6.1 dot_general

**说人话**：就是**广义的矩阵乘法**。左边一个张量、右边一个张量，在指定的一些维上做“乘加”（内积），这些维会**消失**；其余维保留，左边保留左边的、右边保留右边的，拼成结果的 shape。

**先想最熟的情况**：矩阵乘。左矩阵 (M×K)，右矩阵 (K×N)，相乘得 (M×N)。K 这一维在计算时是“逐元素乘再沿 K 求和”，所以结果里**没有 K**，只有 M 和 N。  
`dot_general` 干的就是这件事，只是维数可以更多，且要你明确告诉它：**哪几维是“批”（只对齐、不参与乘加）、哪几维是“收缩维”（做乘加、会消失）、剩下的维谁来自左边、谁来自右边**。

**三种维（由 `dot_dimension_numbers` 指定）**：

1. **Batch 维（batching dimensions）**  
   左边和右边各指定一些维，一一对应，且长度相等。这些维**不参与乘加**，只是“按格对齐”：第 b 格左边和第 b 格右边做一次矩阵乘。结果里会保留这些维。  
   _例_：批量矩阵乘，lhs [B,M,K]、rhs [B,K,N]，把第 0 维标成 batch，则 B 对齐，每一格做 (M×K)@(K×N)→(M×N)，结果 [B,M,N]。

2. **收缩维（contracting dimensions）**  
   左边和右边各指定一些维，一一对应，长度相等。在这些维上做**乘加（内积）**，所以它们**不会出现在结果里**。  
   _例_：上面里的 K：lhs 的第 2 维和 rhs 的第 1 维配成一对 contracting，乘加后 K 消失。

3. **结果维（result dimensions）**  
   除 batch 和 contracting 外，左边剩下的维都来自 lhs，右边剩下的维都来自 rhs；结果里**先放 batch 维，再放 lhs 的独有维，再放 rhs 的独有维**。  
   _例_：lhs [B,M,K] 的 M、rhs [B,K,N] 的 N，结果就是 [B, M, N]。

**结果 shape 小结**：结果 =（batch 维的长度）+（lhs 独有维的长度）+（rhs 独有维的长度）；contracting 维不出现。  
**约束**：lhs/rhs 的 batch 维、contracting 维彼此长度对应相等；元素类型一致；`precision_config` 长度为 2。

### 6.2 reduce

**作用**：在指定 **dimensions** 上，用 **body**（二元计算）对 **inputs** 做归约，**init_values** 为初值。

**形式化**（与规范一致）：

- 所有 `inputs` shape 相同；`results[k]` 的 shape = 从 `inputs[k]` 的 shape 中**去掉** `dimensions` 中的维。
- 对每个 result 的索引，对应 input 上的一块“沿 dimensions 的切片”，用 **body** 从 **init_values** 开始归约（顺序由实现定义），得到该 result 元素。
- **body** 类型：`(tensor<E>, tensor<E>, ...) -> (tensor<E>, ...)`，E 为 input 元素类型；输入输出一一对应 inputs/init_values/results。

**约束**：

- `size(inputs) = size(init_values) = size(results)`；
- 各 input 与对应 init_value、result 元素类型一致；
- `dimensions` 在 `[0, rank(inputs[0]))` 内且互不相同。

**典型**：`reduce(..., dimensions=[1], body=add)` 表示在维度 1 上求和；若只在一个标量维上归约，该维消失。

**例子**（在维度 1 上求和）：

- `input` shape = `[2, 3]`，例如  
  `[[1, 2, 3], [4, 5, 6]]`
- `dimensions = [1]`，在“列”这一维上归约；`init_values = [0]`，`body = add`。
- 结果 shape = 去掉维度 1 → `[2]`。  
  对第 0 行切片 `[1,2,3]` 做 0+1+2+3 = **6**；对第 1 行切片 `[4,5,6]` 做 0+4+5+6 = **15**。  
  故 `result = [6, 15]`。

**多输入例子**（同时求每行的和、以及每行的平方和）：

- **`inputs` 是什么**：`inputs` 是一个**列表**，列表里每一项是一个 tensor。`inputs = [x, x]` 表示「列表长度为 2：第 0 项是矩阵 x，第 1 项也是矩阵 x」-- 也就是同一个矩阵 x 既当第一路输入、又当第二路输入（第一路用来累加「和」，第二路用来累加「平方和」）。不是「一个 2 维向量」。
- 还是用矩阵 `x = [[1,2,3], [4,5,6]]`，`init_values = [0, 0]`，`dimensions = [1]`。我们要得到两个结果：`results[0]` = 每行的和，`results[1]` = 每行的平方和。
- **body 和矩阵的关系**：对「第一行」这一条切片 `[1, 2, 3]`，沿维度 1 会依次取 3 个元素，body 被调用 3 次：
  - 第 1 次：当前累加值 = (0, 0)，当前取到的元素：从 inputs[0] 取到 1，从 inputs[1] 取到 1（因为两路都是 x，同一位置都是 1）→ body(0, 0, 1, 1) 返回 (0+1, 0+1²) = (1, 1)；
  - 第 2 次：当前累加值 = (1, 1)，当前元素 = (2, 2) → 返回 (1+2, 1+4) = (3, 5)；
  - 第 3 次：当前累加值 = (3, 5)，当前元素 = (3, 3) → 返回 (3+3, 5+9) = (6, 14)。
    所以第一行得到 `results[0][0]=6`，`results[1][0]=14`。同理第二行得到 15 和 77。
- **body 参数顺序**：body(累加值1, 累加值2, 当前来自 inputs[0] 的元素, 当前来自 inputs[1] 的元素) → 返回 (新累加值1, 新累加值2)。因为这里两路都是同一个 x，所以第三、第四个参数在每一步都相同。

### 6.3 dynamic_slice / dynamic_update_slice

二者都是“下标在运行时确定”的切片/更新，理解时抓住：**谁决定起始、谁决定形状、越界如何 clamp**。

---

**dynamic_slice**：从 `operand` 里切出一块子张量。**起始位置** = `start_indices`（每个维度一个 0 维 tensor，运行时值），**切出来的形状** = 属性 `slice_sizes`（静态）。若起始越界，会 **clamp** 到合法范围再切。

**例子**：

- `operand` shape `[3, 4]`，例如  
  `[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`
- `start_indices = [1, 2]`（标量 tensor，运行时给出）：从第 1 行、第 2 列开始切。
- `slice_sizes = [2, 2]`（属性）：切出 2×2 的一块。
- 结果：从 (1,2) 开始的 2×2 子块 = `[[7, 8], [11, 12]]`，shape `[2, 2]`。

**越界怎么处理**：对每一维 `i`，合法的起始下标范围是 `[0, dim(operand, i) - slice_sizes[i]]`（保证 start + size 不超出边界）。`start_indices[i]` 若小于 0 则视为 0，若大于该上界则视为上界；再用 clamp 后的起始和 `slice_sizes` 去切，**结果 shape 始终等于 slice_sizes**。

上例中 `operand` shape `[3, 4]`，`slice_sizes = [2, 2]`：维 0 合法起始 = `[0, 3-2]` 即 `[0, 1]`，维 1 合法起始 = `[0, 4-2]` 即 `[0, 2]`。若运行时 `start_indices = [2, 3]`，则维 0 的 2 clamp 成 1，维 1 的 3 clamp 成 2，得到有效起始 `[1, 2]`，切出的仍是 2×2 的一块 `[[7, 8], [11, 12]]`。

---

**dynamic_update_slice**：在 `operand` 的**起始位置** `start_indices` 处，用 `update` 覆盖一块；`update` 的 shape 不能超过 operand 对应范围，越界时起始同样会 **clamp**。

**例子**：

- `operand` 同上：`[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]`，shape `[3, 4]`。
- `start_indices = [1, 1]`：从 (1,1) 开始写。
- `update` shape `[2, 2]`，例如 `[[0, 0], [0, 0]]`。
- 结果：operand 在 (1,1)～(2,2) 这一块被换成 update，其余不变：  
  `[[1, 2, 3, 4], [5, 0, 0, 8], [9, 0, 0, 12]]`。

### 6.4 dynamic_broadcast_in_dim

- 与 **broadcast_in_dim** 的映射规则相同，区别在于 **result 的 shape 由运行时操作数提供**（如 `result_shape` tensor），用于 dynamic batch / dynamic shape 场景。
- 类型推断：结果类型不能仅从操作数静态推断，需在 IR 中显式或由上层推导。

### 6.5 broadcast_in_dim（小结）

- 已在上文 “3.2 broadcast_in_dim” 详述：**broadcast_dimensions** 表示 operand 各维对应 result 的哪一维；大小为 1 的维在结果中复制。
- 与 **reshape** 的区别：broadcast 会**复制数据**、可增加元素个数；reshape 只改 view，元素个数不变。

### 6.6 reshape 与 bitcast_convert 的区别

**reshape**：

- **语义**：保持**元素类型**与**元素个数**不变，只改变 shape（即“同一线性存储”的另一种多维索引解释）。
- 形式化：`result` 与 `operand` 的索引在**字典序**下逐一对应同一元素。
- 约束：`operand` 与 `result` 的 **element type 相同**、**元素总数相同**。

**bitcast_convert**：

- **语义**：不改变底层**比特**，按 **result 的 element type** 重新解释；可能改变元素个数和 layout。
- 若 `num_bits(result_element) < num_bits(operand_element)`：最后一维会“拆开”，`dim(result, R) = num_bits(E)/num_bits(E')`，rank 增加 1。
- 若 `num_bits(result_element) > num_bits(operand_element)`：最后一维会“合并”，`dim(operand, R-1) = num_bits(E')/num_bits(E)`，rank 减少 1。
- 若比特数相等：shape 不变，仅类型重新解释。

**对比**：

|          | reshape              | bitcast_convert          |
| -------- | -------------------- | ------------------------ |
| 元素类型 | 不变                 | 可变                     |
| 比特布局 | 不变（同一线性存储） | 不变（按新类型解释）     |
| 元素个数 | 不变                 | 可变（与类型、维数相关） |
| 典型用途 | 改维数/维大小        | f32↔i32、f32↔4×i8 等     |

---

## 七、小结：读图时建议

1. **先看 type**：各 op 的 operand/result 的 shape、rank、是否含 `?`，再结合 op 语义推断数据流。
2. **broadcast**：找 `broadcast_in_dim` / `dynamic_broadcast_in_dim`，看 `broadcast_dimensions` 和 result shape，理解“谁被复制、映射到哪几维”。
3. **reduce**：看 `dimensions` 与 `body`，明确“在哪些维上、用什么运算”归约，结果 shape 是去掉这些维。
4. **dot_general**：区分 batching、contracting、lhs/rhs result 维，结果 shape = batch 维 + lhs 自由维 + rhs 自由维。
5. **dynamic**：凡 `dynamic_*` 都至少有一个“运行时才知的量”（起始、shape 等），类型上常见 `?` 或 SSA 依赖。
6. **reshape vs bitcast**：改 shape 且元素类型不变、个数不变 → reshape；按新类型重解释比特、可能改维数/个数 → bitcast_convert。
7. **副作用**：有 token、collective、infeed/outfeed 的路径会约束执行顺序，读图时注意依赖关系。

掌握以上语义后，即可在不写 Pass 的前提下，稳定地**阅读、推导和调试** StableHLO/MHLO 图级 IR。

---

## 参考资料

- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)（规范正文）
- [Dynamism in StableHLO](https://openxla.org/stablehlo/dynamism)（动态 shape、refinement pipeline）
- [StableHLO Specification Checklist](https://openxla.org/stablehlo/spec_checklist)（各 op 的 spec/verification/interpreter 状态）
- 本仓库《MLIR端到端实战指南(CPU).md》中关于 StableHLO 构建与工具链的说明
