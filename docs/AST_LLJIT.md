# AST LLJIT 实现

## Lexer & Parser & AST

1. **Lexer** 提供 `nextToken()` 方法。关键字段：`pos`。
2. **Parser** 提供 `parseProgram()`，实质是循环调用 `parseStatement()` 方法，生成 `Vector<Stmt>`。关键字段：`lexer` 和 `curToken`。
3. `parseStatement()` 方法其实就是 **LL(1)**，只向前看一个 token，就能决定语法规则该怎么走。
4. 当局部语法出现 FIRST/FOLLOW 集合冲突时，可以采用**部分 LL(2)** 的方式解决。
5. **Stmt** 提供 `void exec(std::shared_ptr<Env> env)`。
6. **Env** 保存 `name->value` 的 map，并存有指向父 Env 的指针，其提供 `get` / `set` / `has` 等方法。
7. Env 通常保留 assignment 的值（包括匿名函数），以及对于函数调用表达式的形参和实参的映射。

## LLVM Module & LLVMContext & LLJIT

1. 对于函数的 IR，只需要通过 **LLVMContext** 获取 **Builder**，然后通过 Builder 插入 Entry、Body Statement、Return。
2. **Module** 是以文件为单位，每个函数都关联一个 Module。
3. **LLJIT** 可以通过 `LLJITBuilder` 实例创建，然后调用 `addIRModule(ThreadSafeModule(Module, LLVMContext))`，再通过 `lookup("funcname")` 就可以获取函数指针，然后就可以运行。
