# V4 Statement FIRST/FOLLOW 集合冲突分析

## 一、从代码抽象出的语法（BNF）

为简化，记：

- **关键字 token**：`let` `const` `fn` `if` `while` `print` `return` `break` `continue`
- **符号**：`{` `}` `(` `)` `;` `,` `=` `+=` `-=` `*=` `/=`
- **id** 表示标识符
- **Expr** 表示表达式（由 `parseExpression()` 处理）

### 1. Statement 总体结构

```
Statement →
      VarDecl
    | FunctionDecl
    | IfStmt
    | WhileStmt
    | PrintStmt
    | ReturnStmt
    | BreakStmt
    | ContinueStmt
    | Block
    | AssignOrExprStmt
    | ExprStmt
```

### 2. 各种语句（简化）

```
VarDecl        → ("let" | "const") ...
FunctionDecl   → "fn" ...
IfStmt         → "if" ...
WhileStmt      → "while" ...
PrintStmt      → "print" ...
ReturnStmt     → "return" ...
BreakStmt      → "break" ";"
ContinueStmt   → "continue" ";"
Block          → "{" { Statement } "}"
```

### 3. ID 开头的分支（最关键）

```
AssignOrExprStmt →
      id AssignmentTail
    | id CallTail ";"
    | id ";"
```

**Assignment**

```
AssignmentTail →
      AssignOp Expr ";"

AssignOp → "=" | "+=" | "-=" | "*=" | "/="
```

**函数调用**

```
CallTail →
      "(" [ ArgList ] ")"

ArgList →
      Expr { "," Expr }
```

### 4. 普通表达式语句

```
ExprStmt →
      Expr ";"
```

## 二、FIRST 集合

记：

**FIRST(X)** 表示 X 推导的第一个终结符集合

### 各产生式 FIRST

#### Statement

```
FIRST(Statement) =
{
  let, const,
  fn,
  if,
  while,
  print,
  return,
  break,
  continue,
  "{",
  id,
  FIRST(Expr)
}
```

⚠ **注意**：`FIRST(Expr)` 通常也包含 `id`, `number`, `'('` 等

#### AssignOrExprStmt

```
FIRST(AssignOrExprStmt) = { id }
```

#### AssignmentTail

```
FIRST(AssignmentTail) = { "=", "+=", "-=", "*=", "/=" }
```

#### CallTail

```
FIRST(CallTail) = { "(" }
```

#### ExprStmt

```
FIRST(ExprStmt) = FIRST(Expr)
```

## 三、FOLLOW 集合（关键部分）

我们只关心 Statement 相关：

```
FOLLOW(Statement) = { "}", EOF }
```

其它：

```
FOLLOW(AssignOrExprStmt) = FOLLOW(Statement)
FOLLOW(ExprStmt)        = FOLLOW(Statement)
```

## 四、是否存在 FIRST/FIRST 冲突？

重点在这里：

```
Statement →
   ...
 | AssignOrExprStmt
 | ExprStmt
```

对比：

| 产生式 | FIRST |
|--------|-------|
| AssignOrExprStmt | `{ id }` |
| ExprStmt | `FIRST(Expr)` |

而：

```
FIRST(Expr) 通常包含 { id, number, "(", ... }
```

所以：

```
FIRST(AssignOrExprStmt) ∩ FIRST(ExprStmt) = { id }
```

❌ **发生 FIRST/FIRST 冲突**

## 五、为什么代码能工作？

因为你写的是：

**递归下降 + 手工 lookahead + 特判**

代码逻辑是：

```python
if curTok == ID:
    先读 ID
    再看下一个 token：
        如果是 =, +=, ... → assignment
        如果是 (           → function call
        否则               → 变量表达式语句
```

这等价于使用 **LL(2)** 或带回溯的解析，而不是纯 **LL(1)**。

## 六、形式化消除冲突（改写文法）

可以左因子化：

```
Statement →
      ...其他...
    | id StatementIdTail
    | ExprNotStartingWithId ";"

StatementIdTail →
      AssignOp Expr ";"
    | "(" [ ArgList ] ")" ";"
    | ";"
```

这样：

```
FIRST(StatementIdTail) = { "=", "+=", "-=", "*=", "/=", "(", ";" }
```

互不冲突 ✅

## 七、总结

| 项目 | 结论 |
|------|------|
| 是否 LL(1) | ❌ 不是 |
| 冲突类型 | FIRST/FIRST |
| 冲突位置 | `id` 开头：赋值 vs 表达式 |
| 当前实现 | LL(2) / 手工判别 |
| 是否合理 | ✅ 工程上完全合理 |

## 八、一句话评价你的设计

你现在的解析器属于：

**Pratt/递归下降 + 局部二次前瞻的混合 LL 解析器**

这是编译器中非常常见、实用、可维护的做法 👍

Clang、Lua、Python 解析器都有类似结构。
