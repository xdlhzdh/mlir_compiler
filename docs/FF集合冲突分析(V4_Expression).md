# V4 FIRST/FOLLOW 集合冲突分析

## 一、由代码推导出的文法（EBNF → BNF）

为了清晰，我先用接近代码结构的 EBNF：

```
Expression     → Ternary

Ternary        → LogicalOr
               | LogicalOr "?" Expression ":" Ternary

LogicalOr      → LogicalAnd { "||" LogicalAnd }

LogicalAnd     → Equality { "&&" Equality }

Equality       → Relational { ("==" | "!=") Relational }

Relational     → Additive { ("<" | ">" | "<=" | ">=") Additive }

Additive       → Multiplicative { ("+" | "-") Multiplicative }

Multiplicative → Unary { ("*" | "/" | "%") Unary }

Unary          → ("-" | "!") Unary
               | "typeof" Unary
               | Postfix

Postfix        → Primary { "(" [ Expression { "," Expression } ] ")" }

Primary        → INT
               | DOUBLE
               | STRING
               | "true"
               | "false"
               | IDENTIFIER
               | "(" Expression ")"
```

### 改写成纯 BNF（适合分析 FIRST/FOLLOW）

引入辅助非终结符：

```
Expression → Ternary

Ternary → LogicalOr Ternary'

Ternary' → "?" Expression ":" Ternary
         | ε

LogicalOr → LogicalAnd LogicalOr'

LogicalOr' → "||" LogicalAnd LogicalOr'
           | ε

LogicalAnd → Equality LogicalAnd'

LogicalAnd' → "&&" Equality LogicalAnd'
            | ε

Equality → Relational Equality'

Equality' → ("==" | "!=") Relational Equality'
          | ε

Relational → Additive Relational'

Relational' → ("<" | ">" | "<=" | ">=") Additive Relational'
            | ε

Additive → Multiplicative Additive'

Additive' → ("+" | "-") Multiplicative Additive'
          | ε

Multiplicative → Unary Multiplicative'

Multiplicative' → ("*" | "/" | "%") Unary Multiplicative'
                | ε

Unary → "-" Unary
      | "!" Unary
      | "typeof" Unary
      | Postfix

Postfix → Primary Postfix'

Postfix' → "(" ArgList ")" Postfix'
         | ε

ArgList → Expression ArgList'
        | ε

ArgList' → "," Expression ArgList'
         | ε

Primary → INT
        | DOUBLE
        | STRING
        | "true"
        | "false"
        | IDENTIFIER
        | "(" Expression ")"
```

## 二、FIRST 集合计算（核心部分）

我们从底向上：

### FIRST(Primary)

```
FIRST(Primary) = {
  INT, DOUBLE, STRING,
  true, false,
  IDENTIFIER,
  "("
}
```

### FIRST(Postfix)

```
Postfix → Primary Postfix'

FIRST(Postfix) = FIRST(Primary)
```

### FIRST(Unary)

```
Unary → "-" Unary | "!" Unary | "typeof" Unary | Postfix

FIRST(Unary) = { "-", "!", "typeof" } ∪ FIRST(Postfix)
             = { "-", "!", "typeof",
                 INT, DOUBLE, STRING,
                 true, false,
                 IDENTIFIER, "(" }
```

### 逐层向上（都一样）

因为都是：

```
X → Y X'
```

所以：

- FIRST(Multiplicative)
- FIRST(Additive)
- FIRST(Relational)
- FIRST(Equality)
- FIRST(LogicalAnd)
- FIRST(LogicalOr)
- FIRST(Ternary)
- FIRST(Expression)

全部等于：

```
{
  "-", "!", "typeof",
  INT, DOUBLE, STRING,
  true, false,
  IDENTIFIER, "("
}
```

## 三、FOLLOW 集合（关键非终结符）

只列重要的：

### FOLLOW(Expression)

Expression 出现于：

- `( Expression )`
- `? Expression :`
- 参数列表

所以：

```
FOLLOW(Expression) ⊇ { ")", ",", ":" }
```

如果是顶层：

```
FOLLOW(Expression) ⊇ { EOF }
```

### FOLLOW(Ternary)

```
Ternary = Expression

FOLLOW(Ternary) = FOLLOW(Expression)
```

### FOLLOW(LogicalOr)

在：

```
Ternary → LogicalOr Ternary'
```

所以：

```
FOLLOW(LogicalOr) ⊇ FIRST(Ternary') - {ε} = { "?" }
```

并且：

```
FOLLOW(LogicalOr) ⊇ FOLLOW(Ternary)
```

### FOLLOW(Unary)

Unary 在：

```
Multiplicative → Unary Multiplicative'
```

所以：

```
FOLLOW(Unary) ⊇ FIRST(Multiplicative') - {ε}
             = { "*", "/", "%" }
```

如果 ε：

```
FOLLOW(Unary) ⊇ FOLLOW(Multiplicative)
```

层层传播：

```
+, -, <, >, <=, >=, ==, !=, &&, ||, ?, ), ,, :
```

## 四、是否存在 FIRST/FIRST 或 FIRST/FOLLOW 冲突？

我们检查几个关键产生式：

### 1. Unary

```
Unary → "-" Unary
      | "!" Unary
      | "typeof" Unary
      | Postfix
```

FIRST 集：

- `"-"`
- `"!"`
- `"typeof"`
- `FIRST(Postfix) = {INT, DOUBLE, ..., "("}`

👉 完全不相交 ✅

### 2. Ternary'

```
Ternary' → "?" Expression ":" Ternary
         | ε
```

检查：

```
FIRST("?"...) = { "?" }
FOLLOW(Ternary') = FOLLOW(Ternary)
                 = { ")", ",", ":", EOF, ... }
```

`"?" ∉ FOLLOW(Ternary')` ✅

### 3. 所有 X'

全部形式：

```
X' → operator Y X' | ε
```

例如：

```
Additive' → "+" ... | "-" ... | ε
```

FIRST:

```
{ "+", "-" }
```

FOLLOW(Additive') 来自上层：

```
<, >, <=, >=, ==, !=, &&, ||, ?, ), ,, :
```

无交集 ✅

## 五、结论

✅ **表达式文法是严格 LL(1) 的：**

- 无 FIRST/FIRST 冲突
- 无 FIRST/FOLLOW 冲突
- 适合递归下降

✅ **优先级与结合性正确：**

```
一元 > 乘除 > 加减 > 关系 > 相等 > && > || > 三目
```
