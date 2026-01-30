# interpreter_v4：从 AST 到 LLVM IR 完整说明

本文档面向零基础读者，解释 `interpreter_v4.hpp` 中 **USE_LLVM_CODEGEN** 相关代码是如何把**抽象语法树（AST）**转换成 **LLVM IR** 的。

---

## 一、先搞清楚几个概念

### 1.1 什么是 AST（抽象语法树）？

你的源代码是一串文本，例如：

```text
let x = 1 + 2;
print x;
```

编译器/解释器会先把这串文本**解析**成一种树形结构，这就是 **AST**。树上的每个节点代表一种“语法成分”：

- **表达式**：字面量、变量、加减乘除、函数调用等
- **语句**：变量声明、赋值、if、while、return、函数定义等

在 `interpreter_v4.hpp` 里，这些节点就是各种 `Expr` 和 `Stmt` 的子类（如 `LiteralExpr`、`VarExpr`、`BinaryExpr`、`IfStmt`、`WhileStmt` 等）。

### 1.2 什么是 LLVM IR？

**LLVM IR** 是 LLVM 的中间表示（Intermediate Representation），可以理解为一种“接近机器、但仍是人可读”的指令形式。它和具体 CPU 无关，可以被进一步优化、再生成各种目标机器的机器码。

特点简要了解即可：

- 使用 **SSA 形式**：每个变量只赋值一次，便于优化
- **类型明确**：例如 `i32`（32 位整数）、`double`（双精度浮点）、`i1`（1 位，用于布尔）
- 由 **基本块（BasicBlock）** 和 **指令** 组成，控制流用 `br`、`ret` 等跳转

### 1.3 “AST 转 LLVM IR”在做什么？

就是把“树上的节点”逐个翻译成“IR 里的指令和基本块”：

- **表达式节点** → 生成一个或多个 IR 指令，得到一个 `Value*`（代表这个表达式的“值”）
- **语句节点** → 生成控制流（分支、循环、返回）和副作用（存变量、调函数等）

整棵 AST 遍历完，就得到一份完整的 LLVM IR，可以交给 LLVM 去优化、生成机器码或 JIT 执行。

### 1.4 基本块创建与分支指令（CodeGen 常用 API）

读 WhileStmt/IfStmt 的 codegen 时经常会看到 `BasicBlock::Create`、`CreateBr`、`CreateCondBr`，含义如下。

#### BasicBlock::Create(context, name, parent)

```cpp
llvm::BasicBlock *condBB =
    llvm::BasicBlock::Create(ctx.context, "while.cond", fn);
```

- **第一个参数** `context`：`LLVMContext&`，类型等都要和它绑定，一般用 `ctx.context`。
- **第二个参数** `name`：**基本块在 IR 里的名字**（字符串）。可以“随便写”，只要你自己能看懂即可；它只影响**打印出来的 IR 可读性**（例如 `; <label>:42: while.cond`），**不参与语义**。习惯上会写 `"while.cond"`、`"if.then"`、`"while.exit"` 这类有含义的名字，方便调试。
- **第三个参数** `parent`：**这个块所属的 LLVM 函数**（`Function*`）。基本块必须挂在某个函数里，所以创建时要指明“属于哪个函数”；代码里一般用 `ctx.builder.GetInsertBlock()->getParent()` 得到当前正在生成的函数 `fn`，把新块挂到同一个 `fn` 下。

#### CreateBr(dest) 与 CreateCondBr(cond, trueDest, falseDest)

两者都是 **终止指令**：会结束当前基本块，控制流跳到目标块。

| API | 含义 | 生成的 IR 形式 |
|-----|------|----------------|
| **CreateBr(dest)** | **无条件跳转**：总是跳到 `dest` 块 | `br label %dest` |
| **CreateCondBr(cond, trueDest, falseDest)** | **条件跳转**：若 `cond` 为真跳到 `trueDest`，否则跳到 `falseDest`；`cond` 必须是 **i1** 类型 | `br i1 %cond, label %trueDest, label %falseDest` |

- **CreateBr**：用于“无论如何都要去某个块”的情况，例如：进入 while 时先跳到条件块（`Br(condBB)`）、body 执行完回到条件（`Br(condBB)`）、break 跳到出口（`Br(exitBB)`）、if 的 then 块末尾跳到 merge（`Br(mergeBB)`）。
- **CreateCondBr**：用于“根据条件二选一”的情况，例如：if 的条件分支（`CondBr(cond, thenBB, elseBB)`）、while 的条件分支（`CondBr(cond, bodyBB, exitBB)`）。

每个基本块必须以**恰好一条**终止指令（`br`、`ret` 等）结束，所以若 body 里已有 return/break/continue 产生了终止指令，就不要再补一条 `CreateBr`，否则会违反 IR 规则。

#### SetInsertPoint(BB)：插入点

**IRBuilder** 有一个“当前插入点”：之后所有 `CreateXXX`（CreateBr、CreateCondBr、CreateAdd、CreateStore 等）生成的指令，都会**插到这个位置**。

- **SetInsertPoint(BasicBlock *BB)**：把插入点设到块 `BB` 的**末尾**。之后生成的指令都会追加到 `BB` 里，直到再次调用 `SetInsertPoint` 或插入了终止指令（`br`/`ret`）结束当前块。
- 常见用法：
  - 创建完 `condBB` 后 **SetInsertPoint(condBB)**，再对 condition 做 codegen、最后 **CreateCondBr(...)**，这样“条件计算 + 条件分支”就都落在 condBB 里。
  - 创建完 `bodyBB` 后 **SetInsertPoint(bodyBB)**，再 **body->codegen(ctx)**，这样循环体的 IR 都落在 bodyBB。
  - if/while 结束后 **SetInsertPoint(mergeBB)** 或 **SetInsertPoint(exitBB)**，这样“下一条语句”的 IR 会接在合并块/出口块之后。

**要点**：你**必须**在生成新块的指令之前调用 **SetInsertPoint(新块)**，否则指令会继续插到“上一个块”里，控制流就乱了。例如 WhileStmt 里先 `CreateBr(condBB)` 结束当前块，紧接着就要 `SetInsertPoint(condBB)`，再在 condBB 里生成条件相关指令。

### 1.5 算术、内存与比较指令（CreateLoad / CreateStore / CreateFAdd / CreateFCmp 等）

表达式和赋值会生成“算数、读内存、写内存、比较”等指令，常见 API 如下。

#### CreateLoad(ty, ptr, name) 与 CreateStore(value, ptr)

- **CreateLoad(ty, ptr, name)**：从指针 `ptr`（通常是 **AllocaInst**）**读**出类型为 `ty` 的值，返回一个 **Value***。例如变量 `x` 的 codegen：`ctx.builder.CreateLoad(ctx.doubleTy(), alloca, name)`，对应 IR 里 `%x = load double, ptr %x.alloca`。
- **CreateStore(value, ptr)**：把 **value** **写**入指针 **ptr**。例如赋值 `x = 1.0`：先对右侧 codegen 得到 value，再 `CreateStore(value, alloca)`，对应 IR 里 `store double %val, ptr %x.alloca`。

我们的约定里变量都是 **alloca double**，所以 Load 用 `doubleTy()`，Store 的 value 也应是 double（或先转成 double）。

#### 浮点运算：CreateFAdd / CreateFSub / CreateFMul / CreateFDiv

当前数值在 IR 里统一用 **double**，所以算术用 **F** 系列：

| API | 含义 | 典型 IR |
|-----|------|---------|
| **CreateFAdd(L, R, name)** | 浮点加 | `%add = fadd double %L, %R` |
| **CreateFSub(L, R, name)** | 浮点减 | `%sub = fsub double %L, %R` |
| **CreateFMul(L, R, name)** | 浮点乘 | `%mul = fmul double %L, %R` |
| **CreateFDiv(L, R, name)** | 浮点除 | `%div = fdiv double %L, %R` |

参数 L、R 一般为 double 类型的 Value*；最后一个参数是可选的名字，方便读 IR。  
（整数运算对应 **CreateAdd / CreateSub / CreateMul / CreateSDiv** 等，本项目中取模用 **CreateSRem** 配合 FPToSI/SIToFP 做 double→i32→取模→double。）

#### 浮点比较：CreateFCmp(pred, L, R) 与 CreateFCmpONE

- **CreateFCmp(pred, L, R, name)**：按 **谓词 pred** 比较两个 double，得到 **i1**（真/假）。**pred** 是 **llvm::CmpInst::Predicate**，常用浮点谓词：

| 谓词 | 含义 | 源码对应 |
|------|------|----------|
| **FCMP_OEQ** | 有序相等 (L == R) | `==` |
| **FCMP_ONE** | 有序不等 (L != R) | `!=` |
| **FCMP_OLT** | 有序小于 (L < R) | `<` |
| **FCMP_OGT** | 有序大于 (L > R) | `>` |
| **FCMP_OLE** | 有序小于等于 | `<=` |
| **FCMP_OGE** | 有序大于等于 | `>=` |

“有序”表示操作数不是 NaN 时的比较；若需考虑 NaN 有 **FCMP_UEQ/UNE** 等（本项目未用）。

- **CreateFCmpONE(L, R)**：等价于 **CreateFCmp(CmpInst::FCMP_ONE, L, R, ...)**，即 **L != R**，结果为 i1。  
  在本项目里还用来把 **double 当布尔**：`CreateFCmpONE(val, ConstantFP::get(doubleTy(), 0.0))` 表示“val 非零为真”，用于 `while (x)`、`!x`、`x && y` 等需要 i1 的地方。

其他逻辑指令：**CreateAnd / CreateOr / CreateNot** 用于 i1 的与/或/非；**CreateFNeg** 用于浮点取负。

### 1.6 栈与堆在语法层和 IR 层的区别

**语法层面**：本语言（interpreter_v4 支持的语法）**没有**“栈指针”和“堆指针”的区分。用户只写 `let x = 1`、`x = 2`，没有类似 `stack x` / `heap y` 或“指针类型”的语法，栈和堆都是**实现细节**，对语法不可见。

**IR 层面**：

- **类型上不区分**：在 LLVM IR 里，指向栈的指针和指向堆的指针**类型相同**（例如都是 `ptr` 或 `ptr double`）。IR 不标注“这是栈指针 / 这是堆指针”。
- **区别在于指针的来源**：
  - **栈**：由 **alloca** 指令得到。`alloca` 在当前函数的**栈帧**里分配一块空间，返回指向这块空间的指针。本项目的局部变量、参数都是用 **CreateAlloca** + **CreateStore/CreateLoad** 实现的，因此都是**栈上的对象**。
  - **堆**：由**调用外部函数**（如 `malloc`）或 LLVM 的分配相关 intrinsic 得到。返回的指针指向**堆**上分配的内存。本项目当前 **没有** 暴露堆分配（没有 `malloc`/`new` 等），所以生成的 IR 里只有“栈指针”，没有“堆指针”。

总结：语法层没有栈/堆概念；IR 层栈 vs 堆的区分只看**指针从哪来**（alloca → 栈，malloc 等 → 堆），类型本身不区分。

---

## 二、整体流程：从源码到 IR

可以概括成下面这条线：

```text
  源码字符串
       ↓
  Lexer（词法分析）→ Token 流
       ↓
  Parser（语法分析）→ AST（vector<Stmt>）
       ↓
  定义 USE_LLVM_CODEGEN 并包含 interpreter_v4.hpp
       ↓
  generateProgram(module, context, builder, program)
       ↓
  遍历 AST，对每个节点调用 codegen(ctx)
       ↓
  LLVM IR 写入 Module（可打印、优化、编译）
```

也就是说：

1. **Lexer + Parser** 负责：源码 → AST（这部分和“解释执行”共用，与 LLVM 无关）
2. **CodeGen** 只在定义了 `USE_LLVM_CODEGEN` 时编译进来，负责：AST → IR
3. 入口是 **generateProgram**：创建 `main`、先为所有函数生成 IR、再为顶层语句生成 IR

下面所有内容都是在讲“CodeGen 这一步具体怎么做”。

---

## 三、CodeGen 的“工作台”：CodeGenContext

生成 IR 时，需要随时知道：

- 当前在哪个函数里、插指令插到哪一块（**IRBuilder**）
- 变量名对应哪条 **alloca**（符号表）
- 函数名对应哪个 **Function***（函数表）
- 当前在哪个 **while** 里（用于 break/continue）

这些信息都放在一个结构体里，叫 **CodeGenContext**（可以理解为“代码生成的工作台”）。

### 3.1 主要成员（在代码里的含义）

| 成员 | 类型 | 作用（通俗理解） |
|------|------|------------------|
| `builder` | `IRBuilder<> &` | 当前往哪插 IR 指令，所有 `CreateXXX` 都通过它 |
| `module` | `Module *` | 整个 IR 的“容器”，函数、全局常量都挂在这里 |
| `context` | `LLVMContext &` | LLVM 的类型、常量等都要和这个 context 绑定 |
| `namedValues` | `vector<unordered_map<string, AllocaInst*>>` | **符号表**：变量名 → 该变量的 alloca；每一层 map 是一个作用域 |
| `currentFunction` | `Function *` | 当前正在生成的是哪个函数；**仅**在遇到无值 `return;` 时用来选返回类型：在用户函数里用该函数的返回类型（double → `ret double 0.0`），在 main 里为 null 则用 i32（`ret i32 0`）。有值 `return expr;` 不需要它，直接用 expr 的类型。 |
| `functions` | `unordered_map<string, Function*>` | 函数名 → LLVM 的 Function*，用于生成 `call` |
| `loopStack` | `vector<LoopLabels>` | 当前嵌套的 while：每个元素有 `cond`、`exit` 块，供 break/continue 跳转 |

### 3.2 作用域：pushScope / popScope

- **pushScope()**：进入新作用域 → 在 `namedValues` 里压入一层**空 map**
- **popScope()**：离开作用域 → 弹出最内层 map
- **defineAlloca(name, alloca)**：在当前层（`namedValues.back()`）记录“名字 → alloca”
- **findAlloca(name)**：从最内层到最外层查找名字，返回对应的 alloca，找不到返回 `nullptr`

这样就能正确实现“内层变量遮蔽外层同名变量”。

### 3.3 类型辅助函数

- `doubleTy()` → `double` 类型（我们用来表示“数”）
- `i1Ty()` → 1 位整数（表示布尔）
- `i32Ty()` → 32 位整数（main 的返回类型等）
- `i8PtrTy()` → `i8*`（例如给 printf 的格式串用）

---

## 四、类型策略：我们的“约定”

在解释器里，一个变量可以是 int、double、string、bool、function 等；但做 IR 时我们做了**简化**：

- **所有“数值”**（int/double）在 IR 里统一用 **double**
- **布尔** 用 **i1**（0/1）
- **变量**在 IR 里一律是 **alloca double**；布尔要存进变量时，先转成 0.0/1.0 再 store

这样不用在 IR 里做复杂的类型判断，实现简单，且足够覆盖当前支持的语法。

---

## 五、表达式（Expr）如何变成 IR

每个表达式节点都有一个 **codegen(CodeGenContext &ctx)**，返回 **llvm::Value***，表示“这个表达式的值在 IR 里对应哪条指令的结果”。

### 5.1 LiteralExpr（字面量）

- **int / double** → `ConstantFP::get(doubleTy, 值)`，即 IR 里的常量浮点数
- **bool** → `ConstantInt::get(i1Ty, 0 或 1)`
- 其他（如 string）→ 退回成 `0.0` 的常量

### 5.2 VarExpr（变量）

- 用 **findAlloca(name)** 找到该变量在当前作用域链里对应的 **alloca**
- 生成一条 **load**：从该 alloca 里读出 double，返回这个 Value*

### 5.3 BinaryExpr（二元运算）

- 先对 **left**、**right** 分别 **codegen**，得到 L、R
- 若类型是 i1（布尔），先 **UIToFP** 转成 double，再参与运算
- **算术**：`+ - * /` 用 **FAdd / FSub / FMul / FDiv**；`%` 用转成 i32 做 **SRem** 再转回 double
- **比较**：`== != < > <= >=` 用 **FCmp**，得到 i1
- **逻辑**：`&& ||` 先把两边转成 i1（若原是 double 则用 FCmp ONE 与 0 比较），再用 **And / Or**

### 5.4 UnaryExpr（一元运算）

- **-**：先 codegen 得到操作数，若为 i1 先转 double，再 **FNeg**
- **!**：先得到操作数，若为 double 则用 FCmp 转成 i1，再 **Not**

### 5.5 TernaryExpr（三元运算符 `a ? b : c`）

- 对 **condition** codegen，得到 i1（否则用 FCmp 转成 i1）
- 对 **trueExpr**、**falseExpr** 分别 codegen
- 用 **CreateSelect(cond, trueVal, falseVal)** 生成 IR 的 select 指令

### 5.6 CallExpr（函数调用）

- 当前只支持“命名函数”：callee 必须是 **VarExpr**（如 `add(1, 2)` 里的 `add`）
- 用 **getName()** 得到函数名，在 **ctx.functions** 里查对应的 **Function***
- 对每个实参 **codegen**，若结果是 i1 先转成 double，再 **CreateCall(F, argVals)**

匿名函数（closure）在 codegen 里会直接抛错，不生成 IR。

### 5.7 ClosureExpr / TypeofExpr

- **ClosureExpr**：codegen 中直接抛异常，表示“匿名函数暂不支持生成 IR”
- **TypeofExpr**：codegen 返回 nullptr，不生成有用 IR

---

## 六、语句（Stmt）如何变成 IR

语句的 **codegen** 没有返回值（void），作用是：在当前位置插入控制流和副作用（store、call 等）。

### 6.1 ExprStmt（表达式语句）

- 对内部的 **expr** 调用 **codegen**，忽略返回值（仅为了求值/副作用，如函数调用）。

### 6.2 BlockStmt（块 `{ ... }`）

- **pushScope()**
- 对块内每一条 **stmt** 依次 **codegen**
- **popScope()**

这样块内的变量不会影响到块外。

### 6.3 VarDeclStmt（变量声明 `let x = ...`）

- **CreateAlloca(doubleTy)** 为变量分配“槽位”
- **defineAlloca(name, alloca)** 把名字登记到当前作用域
- 若有初值，对 **init** codegen，若得到 i1 先转 double，再 **Store** 到 alloca；否则 store 0.0

### 6.4 AssignStmt（赋值 `x = ...` 或 `x += ...`）

- **findAlloca(name)** 找到变量的 alloca
- 对右侧 **value** codegen，得到 newVal（i1 则先转 double）
- **=**：直接 **Store(newVal, alloca)**
- **+= -= *= /=**：先 **Load** 旧值，再 FAdd/FSub/FMul/FDiv，最后 **Store** 结果

### 6.5 PrintStmt（`print a, b;`）

- 若 module 里还没有 **printf**，先声明一个：`int printf(i8*, ...)`
- 对每个表达式 codegen，若为 i1 转成 double，用格式串 `"%g "` 或 `"%g\n"` 调用 **printf**

（当前只支持数值打印，字符串未在 codegen 中实现。）

### 6.6 IfStmt（`if (cond) thenStmt else elseStmt`）

- 对 **condition** codegen，得到 i1（否则 FCmp 转成 i1）
- 创建 3 个基本块：**thenBB**、**elseBB**（若有 else）、**mergeBB**
- **CondBr(cond, thenBB, elseBB 或 mergeBB)**
- 在 **thenBB** 里对 thenStmt codegen；若当前块还没有**终止指令**（如 return），再 **Br(mergeBB)**
- 若有 else，在 **elseBB** 里对 elseStmt codegen，同样在无终止指令时 **Br(mergeBB)**
- 最后 **SetInsertPoint(mergeBB)**，后续代码接在“if 之后”

这样就不会在 return 后面再插 br，符合 LLVM“每个块末尾恰好一条终止指令”的规则。

### 6.7 WhileStmt（`while (cond) body`）

#### 6.7.1 实现概览

- 创建 **condBB**、**bodyBB**、**exitBB**（`BasicBlock::Create` 的第二个参数是块名、第三个是所属函数，见 §1.4；**CreateBr** 为无条件跳转，**CreateCondBr** 为条件跳转，亦见 §1.4）
- 先 **Br(condBB)**，再 **SetInsertPoint(condBB)**
- 把 **(condBB, exitBB)** **push** 进 **loopStack**（给 break/continue 用）
- 对 **condition** codegen 得到 i1，**CondBr(cond, bodyBB, exitBB)**
- 在 **bodyBB** 里对 **body** codegen；若当前块还没有终止指令，再 **Br(condBB)** 回到条件
- **pop** loopStack，**SetInsertPoint(exitBB)**

同样，若 body 里有 return/break/continue，就不会再插“回到 cond”的 br，避免违反 IR 规则。

#### 6.7.2 为什么要这么实现？（详细说明）

**1. 为什么需要三个基本块？**

LLVM IR 里控制流由“基本块 + 终止指令”描述。`while (cond) body` 的语义是：

- **先判断 cond**：为真则执行 body 并再次判断，为假则离开循环。
- **body 里**可能有 **break**（跳到循环外）、**continue**（跳到下一轮，即再次判断 cond）、**return**（退出函数）。

因此需要三个块分工明确：

| 块 | 名字 | 作用 |
|----|------|------|
| **condBB** | `while.cond` | 计算条件、根据结果跳到 body 或出口 |
| **bodyBB** | `while.body` | 执行循环体 |
| **exitBB** | `while.exit` | 循环结束后的“汇合点”，后续语句接在这里 |

这样 **break** 跳到 exitBB，**continue** 跳到 condBB，结构清晰且与语义一致。

**2. 为什么先 `CreateBr(condBB)`？**

进入 WhileStmt::codegen 时，**builder 的插入点**还在“上一句语句”所在的块里（例如 main 的某块、或 if 的 then/else 块）。我们要开始这个 while，就必须从当前块**离开**，去第一个该执行的地方——也就是**条件块**。

因此第一步是：**在当前块末尾插一条无条件跳转 `Br(condBB)`**。这样控制流就正式进入“while 的领地”：先到 condBB 判断条件。同时，**SetInsertPoint(condBB)** 把后续要生成的指令都放到 condBB 里（先是一条条条件计算，最后是 CondBr）。

**3. 为什么 cond 要转成 i1？**

LLVM 的 **CondBr** 要求条件类型是 **i1**（1 位整数，表示真/假）。我们的 AST 里条件可能是 double（例如 `while (x)`）或已经是 bool。因此：

- 若 codegen 得到 **nullptr**（少见），用常量 1（真）兜底。
- 若得到的不是 i1（例如 double），用 **FCmpONE(cond, 0.0)** 得到“非零即真”的 i1，再交给 CondBr。

这样无论源码里写的是 `while (x)` 还是 `while (x > 0)`，都能得到合法的 i1 条件。

**4. 为什么用 loopStack 存 (condBB, exitBB)？**

**BreakStmt** 和 **ContinueStmt** 在 codegen 时只知道“要跳出循环”或“要下一轮”，不知道具体跳到哪个块。而一个函数里可能有**嵌套的 while**，所以必须“当前循环”对应哪两个块：

- **break** → 跳到**当前**循环的 **exitBB**
- **continue** → 跳到**当前**循环的 **condBB**（重新判断条件）

因此用 **loopStack**：每进入一个 WhileStmt 就 **push_back({condBB, exitBB})**，离开时 **pop_back()**。Break/Continue 只需 **loopStack.back()** 就能拿到当前循环的 cond 和 exit，实现正确的跳转；嵌套循环时栈顶自然是内层循环的标签。

**5. 为什么 body 后要判断“若当前块没有终止指令再 Br(condBB)”？**

在 LLVM 里，**每个基本块必须以恰好一条终止指令结束**（如 `br`、`ret`）。在 bodyBB 里我们调用了 **body->codegen(ctx)**，body 里可能：

- 普通语句 → 不会产生终止指令，执行完应**回到 condBB** 继续下一轮。
- **return** → 已产生 **CreateRet(...)**，本块已终止。
- **break** → 已产生 **CreateBr(exitBB)**，本块已终止。
- **continue** → 已产生 **CreateBr(condBB)**，本块已终止。

所以只有在 **GetInsertBlock()->getTerminator()** 为空时，才需要补一条 **Br(condBB)**，否则会违反“一块一条终止指令”的规则。

**6. 为什么最后要 pop loopStack 并 SetInsertPoint(exitBB)？**

- **pop**：while 的“作用范围”结束了，内层 break/continue 不应再指向这个循环的 cond/exit；弹出后若还有外层循环，栈顶就变成外层的 LoopLabels。
- **SetInsertPoint(exitBB)**：while 之后的代码（下一条语句）应该接在 **exitBB** 之后。exitBB 此时可能还没有任何指令（只是空块），但插入点设在这里，后续生成的指令就会填到 exitBB，形成“循环出口 → 下一条语句”的正确顺序。

**7. 控制流小结（示意）**

```text
  ... (前面的代码)
       |
       v
  [当前块] ----Br(condBB)---->
       |                          |
       |                          v
       |                    [condBB: while.cond]
       |                    计算 cond，CondBr(cond, bodyBB, exitBB)
       |                    /                    \
       |                   /                      \
       |                  v                        v
       |            [bodyBB: while.body]      [exitBB: while.exit]
       |            body->codegen(ctx)              |
       |            若无终止指令: Br(condBB) ----> 回到 condBB
       |                  |                         |
       |            (return/break/continue 会       |
       |             直接跳到 ret/exit/cond)        v
       |                                    ... (while 后面的代码)
```

总结：WhileStmt 的 codegen 通过 **三个基本块 + 先 Br(condBB) + loopStack + 按需 Br(condBB)**，既满足了 LLVM“每块一条终止指令”的约束，又正确实现了 while、break、continue 的语义，并支持嵌套循环。

### 6.8 ReturnStmt（`return` 或 `return expr`）

- **有值**：对 value codegen，若为 i1 转 double，**CreateRet(v)**；若 codegen 失败则 **CreateRet(0.0)**
- **无值**：根据**当前函数的返回类型**生成“零值”并返回：
  - 若在某个用户函数里（**currentFunction** 非空）：用该函数的 **getReturnType()**，整数则 **ConstantInt::get(..., 0)**，否则 **ConstantFP::get(..., 0.0)**，再 **CreateRet**
  - 若在顶层（currentFunction 为空，即 main）：用 **i32**，**CreateRet(0)**

这样就不会对“返回 double 的函数”或 main 使用 **CreateRetVoid()**，避免非法 IR。

### 6.9 BreakStmt / ContinueStmt

- 从 **loopStack.back()** 取出当前循环的 **exit**（break）或 **cond**（continue）
- **CreateBr(exit)** 或 **CreateBr(cond)**，不再执行循环体/本轮后续代码

### 6.10 FunctionStmt（`fn name(params) { body }`）

- 用 **double** 为每个参数建类型，**FunctionType::get(doubleTy, paramTypes, false)**，再 **Function::Create(..., name, module)**
- 把 **name → F** 写入 **ctx.functions**
- 建 **entry** 基本块，**SetInsertPoint** 到该块
- **pushScope()**
- 对每个参数：**CreateAlloca(double)**，**Store(参数, alloca)**，**defineAlloca(参数名, alloca)**
- **ctx.currentFunction = F**，对 **body** codegen，再 **currentFunction = nullptr**
- 若函数末尾还没有终止指令，**CreateRet(0.0)**
- **popScope()**

这样每个用户函数都对应一个返回 double 的 LLVM 函数，且参数和局部变量都在各自的 alloca 里。

---

## 七、顶层入口：generateProgram

这是“把整棵 AST 变成一整个 Module”的入口，逻辑是：

1. 构造 **CodeGenContext**，**pushScope()** 一层顶层作用域
2. 创建 **main** 函数：**FunctionType::get(i32, false)**，无参，返回 i32；建 **entry** 块，然后 **builder.SetInsertPoint(mainEntry)**，表示“接下来插的指令都进 main 的 entry 块”。
3. **第一遍**：遍历 program，只对 **FunctionStmt** 做 **codegen**  
   - 这样先把所有“用户定义的函数”都生成到 module 里，并登记到 **ctx.functions**  
   - **为何要恢复插入点**：每次调 **FunctionStmt::codegen** 时，builder 的插入点会被改到**该用户函数内部**（生成其 body 的 IR）。所以循环里在 codegen 前 **savedBlock = builder.GetInsertBlock()**（此时是 main 的 entry），codegen 后再 **builder.SetInsertPoint(savedBlock)**，把插入点**还原回 main 的 entry**。否则下一轮循环或之后的指令会误插到上一个用户函数里。
4. **第二遍前**再 **builder.SetInsertPoint(mainEntry)**：明确保证“第二遍插的顶层语句（变量、print、if、while 等）都进 main 的 entry”。虽然第一遍结束时理论上已在 mainEntry，显式设一遍意图更清晰，也避免以后改代码时插入点不在 main 的情况。然后**第二遍**遍历 program，对**非** FunctionStmt 的语句做 **codegen**。
5. 若 main 的当前块还没有终止指令，**CreateRet(0)**（main 返回 i32）
6. **popScope()**

这样得到的 module 里就有：一个 **main**（对应顶层语句）+ 若干用户函数（如 add、fib），且所有名字都能在 codegen 时正确解析。

---

## 八、怎么用（写代码的人）

1. 在包含 `interpreter_v4.hpp` 的**某一个** .cpp 里定义宏（通常放在 include 之前）：
   ```cpp
   #define USE_LLVM_CODEGEN 1
   #include "interpreter_v4.hpp"
   ```
2. 准备好 **LLVMContext**、**Module**、**IRBuilder**（和简单示例里一样）
3. 用 **Parser** 解析源码得到 **program**（`vector<unique_ptr<Stmt>>`）
4. 调用：
   ```cpp
   Interpreter_V4::generateProgram(module.get(), context, builder, program);
   ```
5. 之后可以 **module->print(...)** 打印 IR，或交给 LLVM 做优化、生成机器码、JIT 等

注意：编译该 .cpp 时要链上 LLVM，并且建议加 **-fexceptions**（因为 LLVM 默认 -fno-exceptions，而我们的 parser/codegen 里会抛异常）。

---

## 九、当前限制（小白须知）

- **匿名函数**（closure）不生成 IR，遇到会抛异常
- **print** 对数值用 `%g`、对**字符串字面量**用 `%s` 生成 printf；变量持字符串未实现（变量在 IR 里仍是 double）
- **typeof** 对**字面量**生成常量字符串 IR（"int"/"double"/"string"/"bool" 等），对变量默认返回 "double"
- 数值在 IR 里统一为 double，不区分为 int/double（若以后要区分，需要扩展类型策略和 codegen）

### 为什么会有这些限制？

| 限制 | 原因 |
|------|------|
| **匿名函数** | 解释器里闭包用 `Function(params, body, capturedEnv)` 存捕获环境，调用时用 `callEnv = make_shared<Env>(func->capturedEnv)` 建链。IR 里当前只对**具名函数**生成 LLVM Function，变量/参数全是 double；匿名函数没有“函数值”的 IR 表示（无函数指针、无闭包结构体），且闭包要捕获的变量是当前栈上的 Alloca，不能直接传给另一函数。要支持需要做闭包转换：分配捕获结构体、生成带 (struct*, ...params) 的内部函数等。 |
| **print 字符串** | `PrintStmt::codegen` 里对所有子表达式统一用 `%g`（double）调 printf；且 IR 里变量/字面量目前只有 double/i1，没有 i8* 字符串类型。字面量里 STRING 在 `LiteralExpr::codegen` 里走 default 被当成 0.0。要支持需：字面量 STRING 生成 `CreateGlobalStringPtr`，Print 里根据 `Value*` 类型选 `%s` 或 `%g`；变量持字符串则需扩展 Alloca/类型策略。 |
| **typeof** | 解释器里 `TypeofExpr::eval` 根据 Value 的 type 返回 "int"/"double"/"string" 等。IR 里 `TypeofExpr::codegen` 直接 `return nullptr`，因为当前类型策略是“一切数值即 double”，没有运行时类型信息（无 tagged union），编译期也无法表达“运行时类型”。要支持需：要么对字面量做静态 typeof 返回常量字符串；要么引入带类型标签的值（tagged value）在运行时查 tag。 |
| **数值统一 double** | 设计选择：CodeGenContext 里变量全是 `AllocaInst` 存 double，函数参数/返回值也是 double，实现简单。解释器里 int/double 并存，但 IR 未区分。要区分需：类型传播或推断、Alloca 用不同 llvm::Type、二元运算/返回按类型选 CreateSDiv/CreateFDiv 等。 |

### 可以解决掉吗？

- **匿名函数**：可以，但工作量较大（闭包转换、捕获结构体、函数指针）。
- **print 字符串**：可以；先支持**字符串字面量**（LiteralExpr STRING 生成 i8*，Print 按类型选 %s/%g）即可，变量持字符串需再扩展类型策略。
- **typeof**：可以；先对**字面量**做静态 typeof 返回常量字符串（"int"/"double"/"string"/"bool"），变量可默认返回 "double" 或后续再做运行时类型。
- **int/double 区分**：可以；需扩展类型策略和 codegen（Alloca 类型、运算指令选择等）。

代码中已实现：**print 对字符串字面量的支持**（`LiteralExpr` 的 STRING 生成 `CreateGlobalStringPtr`，`PrintStmt::codegen` 根据 `Value*` 是否为指针选 `%s` 或 `%g`）、**typeof 对字面量生成常量字符串 IR**（`TypeofExpr::codegen` 对 `LiteralExpr` 按类型返回对应字符串全局指针，否则返回 `"double"`）。

---

## 十、小结：一句话版

**AST 转 LLVM IR** = 对 AST 里每个 **Expr** 调用 **codegen** 得到 **Value***，对每个 **Stmt** 调用 **codegen** 在 **CodeGenContext** 的 **builder** 当前位置插入控制流和 store/call；**CodeGenContext** 负责符号表、函数表、循环栈和类型约定，**generateProgram** 负责先生成所有函数再生成 main 的顶层语句，最终得到合法、可被 LLVM 继续处理的 IR。

如果你是从“解释执行”看过来的：解释时用的是 **eval(env)** / **exec(env)**，走的是“树 + 环境”；生成 IR 时用的是 **codegen(ctx)**，走的是“树 + 工作台”，产出的是指令而不是运行时的值。两者共用同一棵 AST，只是“遍历时做的事”不同。
