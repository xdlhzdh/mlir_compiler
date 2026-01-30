# LLVM 优化 Pass 完整总结：mem2reg, instcombine, simplifycfg, sroa

本文档汇总 LLVM 中与 SSA 构建、控制流简化和内存提升相关的核心优化 Pass：`mem2reg`、`instcombine`、`simplifycfg`、`sroa`。它们协同工作，将前端生成的原始 IR 转换为高效、简洁的 SSA（静态单赋值）形式；其中 **sroa** 不仅处理结构体/数组，也处理标量类型的复杂内存操作（如类型双关）。

---

## 1. 核心 Pass 功能速览

| Pass 名称 | 全称 | 操作对象 | 核心作用 | 典型变换 / 行为 |
| --- | --- | --- | --- | --- |
| **mem2reg** | Memory to Register | `alloca` 变量 | **SSA 构造器**。仅处理**简单**的局部变量提升。 | 消除标准 `load/store`，插入 ϕ 节点。 |
| **instcombine** | Instruction Combiner | 指令 (Instructions) | **指令化简**。代数规则与窥孔优化 (Peephole)。 | `add x, 0 -> x`；`fsub x, 1 -> fadd x, -1`；`mul x, 2 -> shl x, 1`。 |
| **simplifycfg** | Simplify CFG | 控制流图 (CFG) | **控制流清理**。简化基本块与跳转逻辑。 | 合并块、删除死代码、统一 `ret` 出口；分支转 `select`。 |
| **sroa** | Scalar Replacement of Aggregates | `alloca` 内存块 | **内存切片与提升**。结构体/数组及**复杂标量**的内存提升。 | 将内存块拆解为寄存器值，插入 `bitcast` 处理类型不匹配。 |

### 1.1 为什么 sroa 叫 “Aggregates” 却处理标量？

虽然名字是 “Scalar Replacement of **Aggregates**”（聚合体标量替换），但在 LLVM 里 sroa 的操作对象本质上是 **`alloca` 指令（栈内存分配）**。它不关心 C 里定义的是 `struct`、`array` 还是 `float`，只关心：**这块内存是如何被使用的**。

- **mem2reg 的局限**：要求 `alloca` 的用途必须是“完美匹配”的 `load`/`store`。若对 `float` 取地址再按 `int*` 读写，`mem2reg` 会放弃（类型不匹配）。
- **sroa 的做法**：把 `alloca` 视为一块“字节缓冲区”。例如对 4 字节的 `float` 按 `i32` 读写时，sroa 可判定这块内存可提升，并插入 `bitcast` 在寄存器中完成类型转换，从而消除内存访问。

因此，**现代 LLVM 流水线中，sroa 实际上承接并扩展了 mem2reg 的工作，处理 mem2reg 处理不了的“脏活累活”（标量类型双关、部分读写等）。**

---

## 2. 深入理解各 Pass 行为

### mem2reg (Memory to Register Promotion)

代码进入 SSA 形式的关键一步。

- **主要任务**：扫描函数中的 `alloca`，若该变量只被简单的 `load`/`store` 使用，则将其删除并提升到虚拟寄存器。
- **ϕ 的产生**：当变量在不同控制流分支中被赋值、并在汇合处被使用时，`mem2reg` 会根据**支配边界 (Dominance Frontier)** 自动插入 ϕ 节点。

### instcombine (Instruction Combiner)

在 SSA 基础上做代数化简与窥孔优化。

- **规范化**：将多种写法统一（如减法统一为加负数），便于后续优化识别。
- **强度折减**：将高开销指令换成低开销指令（如 `mul x, 2 -> shl x, 1`）。
- **元数据标注**：推导并添加属性（如 `nonnull`, `noundef`），为后端提供更多指针/值安全信息。

### simplifycfg (CFG Simplification)

负责调整程序的“骨架”。

- **统一出口 (Unify Returns)**：将函数中多个 `ret` 合并为一个，通常在此处插入 ϕ 节点。
- **分支转选择**：将简单 `if-else` 拍扁成 `select` 指令，减少分支预测压力。
- **清理**：删除不可达的代码块。

### sroa (Scalar Replacement of Aggregates)

处理对象是 **`alloca` 指向的内存块**，既包括 C 中的 `struct`/数组，也包括**被当作缓冲区使用的标量**（如 `float` 强转为 `int` 读写）。

- **标量类型双关**：当 `alloca` 被不同类型访问（如 `float` 存、`i32` 读）时，sroa 可分析出“全覆盖”访问，插入 `bitcast` 在寄存器中完成转换，消除内存访问；mem2reg 在此类情况下会放弃。
- **结构体/数组**：递归拆解聚合体，将各字段或元素提升为独立标量；未使用的字段会被彻底剔除。

---

## 3. 典型协同流程

1. **原始 IR**：局部变量多在堆栈上（`alloca`），控制流复杂、可能存在多处返回。
2. **sroa（现代流水线中常先于或替代 mem2reg）**：分析所有 `alloca`（结构体与标量），对类型双关插入 `bitcast` 提升，对结构体拆解为标量，构建初步 SSA。
3. **mem2reg**：对剩余“简单”的 `alloca` 做标准提升，变量进入寄存器，ϕ 节点补全 SSA。
4. **simplifycfg**：合并 `ret`，清理因提升后变空的基本块；合并路径时可能新增 ϕ 节点。
5. **instcombine**：在简洁的 CFG 基础上，对寄存器运算做数学化简与常量折叠。

---

## 4. 实例：SSA 构建与控制流简化 (mem2reg + simplifycfg)

以斐波那契为例，展示变量提升与多返回合并。

### C 源代码 (fib.c)

```c
double fib(double n) {
    if (n < 2.0) {
        return n;   // 出口 1
    }
    return fib(n - 1.0) + fib(n - 2.0);  // 出口 2
}
```

### 阶段 1：原始 IR（Clang 生成）

*特征：存在 `alloca`，多个 `ret`。*

```llvm
define internal double @fib(double %n) {
entry:
  %n.addr = alloca double, align 8
  store double %n, ptr %n.addr, align 8
  %0 = load double, ptr %n.addr, align 8
  %cmp = fcmp olt double %0, 2.000000e+00
  br i1 %cmp, label %if.then, label %if.end
  ; ...
}
```

### 阶段 2：优化后 IR (mem2reg + simplifycfg)

*特征：`alloca` 消失，`ret` 合并，出现 ϕ 节点。*

```llvm
define internal double @fib(double %n) {
entry:
  %cmp = fcmp olt double %n, 2.000000e+00
  br i1 %cmp, label %common.ret, label %if.end

common.ret:
  %common.ret.op = phi double [ %add, %if.end ], [ %n, %entry ]
  ret double %common.ret.op

if.end:
  %sub = fadd double %n, -1.000000e+00   ; instcombine: fsub x, 1.0 -> fadd x, -1.0
  %call = call double @fib(double %sub)
  ; ...
  br label %common.ret
}
```

---

## 5. 实例：instcombine 指令化简

`instcombine` 在 SSA 基础上对单条或相邻指令做代数化简、强度折减与规范化，不改变控制流。下面用一个小函数展示几种典型变换。

### C 源代码（示意）

```c
int combine_demo(int x, int y) {
    int a = x + 0;           // 加 0 -> 直接使用 x
    int b = a * 2;          // 乘 2 -> 左移 1
    int c = y - 1;          // 减 1 -> 加 -1（有符号）
    return b + c;
}
```

### instcombine 前的 IR（示意）

*特征：存在“冗余”运算，写法未规范化。*

```llvm
define i32 @combine_demo(i32 %x, i32 %y) {
  %a = add i32 %x, 0              ; add x, 0 可消去
  %b = mul i32 %a, 2              ; mul a, 2 -> shl a, 1
  %c = sub i32 %y, 1              ; sub y, 1 -> add y, -1
  %r = add i32 %b, %c
  ret i32 %r
}
```

### instcombine 后的 IR

*特征：加 0 消去、乘 2 变为左移、减 1 规范为加 -1（便于后续优化或后端）。*

```llvm
define i32 @combine_demo(i32 %x, i32 %y) {
  %b = shl i32 %x, 1              ; x + 0 被消去，mul x, 2 -> shl x, 1
  %c = add i32 %y, -1             ; sub y, 1 -> add y, -1
  %r = add i32 %b, %c
  ret i32 %r
}
```

| 原始 IR | instcombine 后 | 说明 |
| --- | --- | --- |
| `add i32 %x, 0` | （消去，直接用 `%x`） | 加 0 恒等 |
| `mul i32 %a, 2` | `shl i32 %x, 1` | 乘 2 强度折减为左移 |
| `sub i32 %y, 1` | `add i32 %y, -1` | 减法规范为加负数 |

浮点同理：`fsub x, 1.0` 会规范成 `fadd x, -1.0`（如 fib 例中的 `fadd double %n, -1.000000e+00`）。此外，`instcombine` 还会根据上下文给指针/值加 `nonnull`、`noundef` 等属性，便于后端优化。

---

## 6. 实例：sroa 进阶（类型双关 + 嵌套结构体）

### 场景 A：标量类型双关 (Type Punning on Scalars)

**疑问**：没有 struct 为什么也能用 sroa？  
**解答**：C 里是 `float f`，在 IR 里是 4 字节的 `alloca`。用 `int*` 访问时，`mem2reg` 会因类型不匹配放弃；sroa 能分析出这是对同一块内存的“全覆盖”访问，插入 `bitcast` 在寄存器中完成转换，消除内存访问。

#### C 源代码

```c
float type_punning() {
    float f = 3.14f;
    int *iptr = (int *)&f;
    *iptr = *iptr & 0x7FFFFFFF;
    return f;
}
```

#### 原始 IR

*特征：`alloca float` 被按 `i32` 存取，mem2reg 无法处理。*

```llvm
  %f = alloca float, align 4
  store float 0x40091EB860000000, ptr %f, align 4
  %0 = load i32, ptr %f, align 4
  %and = and i32 %0, 2147483647
  store i32 %and, ptr %f, align 4
  %1 = load float, ptr %f, align 4
  ret float %1
```

#### sroa 优化后 IR

*特征：内存提升为寄存器，用 `bitcast` 处理类型。*

```llvm
define float @type_punning() {
  %1 = bitcast float 0x40091EB860000000 to i32
  %2 = and i32 %1, 2147483647
  %3 = bitcast i32 %2 to float
  ret float %3
}
```

---

### 场景 B：嵌套结构体 (Nested Structs)

sroa 的本职工作之一：递归拆解聚合体，未使用字段直接剔除。

#### C 源代码

```c
struct Inner { int a; int b; };
struct Outer { struct Inner in; int c; };

int nested(int val) {
    struct Outer out;
    out.in.a = val;
    out.c = 10;
    return out.in.a + out.c;   // out.in.b 未使用
}
```

#### 原始 IR

*特征：通过 `getelementptr` 计算偏移。*

```llvm
  %out = alloca %struct.Outer, align 4
  %in = getelementptr inbounds %struct.Outer, ptr %out, i32 0, i32 0
  %a = getelementptr inbounds %struct.Inner, ptr %in, i32 0, i32 0
  store i32 %val, ptr %a, align 4
  ; ...
```

#### sroa 优化后 IR

*特征：结构体概念消失，`out.in.b` 被剔除。*

```llvm
define i32 @nested(i32 %val) {
entry:
  %add = add nsw i32 %val, 10
  ret i32 %add
}
```

---

## 7. 关键标志解释

### `add nsw` (No Signed Wrap)

- **含义**：加法**不会发生有符号溢出**。
- **来源**：C/C++ 规定有符号整数溢出为未定义行为 (UB)。
- **作用**：允许编译器假设 `a + 1 > a` 恒成立，从而优化掉多余溢出检查；若实际溢出，行为未定义。

### `noundef` 与 `nonnull`

- **noundef**：值不是未定义的 (undef)。
- **nonnull**：指针不为 NULL。
- **作用**：`instcombine` 根据上下文（如全局变量地址、常量）添加这些属性，告知后端可安全访问或省略空指针检查，便于更激进优化。

---

## 8. 常见疑问解答（FAQ）

- **为什么 `mem2reg` 没产生 ϕ？**  
  若分支没有汇合（例如每分支都直接 `ret`），或变量在分支中未被多次修改，则不需要 ϕ 节点。

- **为什么 `simplifycfg` 会产生 ϕ？**  
  将多个 `ret` 合并为一个公共返回块时，为决定最终返回哪个分支的值，必须在汇合处插入 ϕ 节点。

- **为什么出现 `noundef` 和 `nonnull`？**  
  由 `instcombine` 根据分析推导出的属性，表示该值/指针是已定义且非空，便于后端优化。

- **为什么没有 struct 也能用 sroa？**  
  sroa 针对的是 `alloca` 指向的内存块；无论是 `struct`、数组还是标量，只要该块被“可分析”的方式访问（如标量类型双关），sroa 都可能通过 `bitcast` 等提升到寄存器。

---

## 9. DCE 与 `simplifycfg` 的区别

DCE（Dead Code Elimination，死代码消除）与 `simplifycfg` 都会删掉“死”的东西，但侧重点不同。

### simplifycfg：侧重于“路”（Control Flow）

- **主要任务**：简化程序的控制流结构。
- **它删掉的是**：永远走不到的“路口”（基本块）、没有意义的“绕路”（多余跳转），或把两段路合并成一段。
- **举例**：`if (false) { ... }` 会被 `simplifycfg` 整块删掉，本质是通过**控制流分析**实现的。

### DCE：侧重于“物”（Instructions）

- **主要任务**：关注指令的计算结果是否被使用。
- **它删掉的是**：算出来之后没人用的“废品”。
- **举例**：`int a = 5 + 10;` 且后面从未使用 `a`，DCE 会抹除这条加法指令；**不关心跳转逻辑**，只关心指令的依赖链。

---

## 10. 总结：Pass 协作逻辑

1. **sroa（现代流水线中替代/扩展 mem2reg）**
   - 分析所有 `alloca`（结构体与标量）。
   - 对复杂类型重解释（如 float/int 双关）插入 `bitcast` 并提升。
   - 对结构体/数组拆解为标量，构建初步 SSA。

2. **mem2reg**
   - 对剩余“简单”的 `alloca` 做标准 load/store 消除与 ϕ 插入。

3. **simplifycfg**
   - 清理控制流，合并 `ret`，必要时在汇合处生成 ϕ 节点。

4. **instcombine**
   - 在寄存器层面做数学化简、强度折减与属性标注（如 `noundef`、`nonnull`、`add nsw`）。
