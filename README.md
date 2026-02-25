# MLIR Compiler

当前阶段主要介绍基于LLVM的普通编译器的实现，包含多版解释器（V1–V4）、ANTLR 解析、NFA/DFA 与 LLVM IR 生成、Pass优化等。后续将逐步补充 MLIR 相关内容。

## 构建

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## 用法

以下命令均在 **build 目录** 下执行（或使用 `cmake --build build --target <target>` 从项目根目录执行）。  
支持：**run / run_ast / run_pass**、**run_simple_pass / run_remove_trivial_block / run_remove_trivial_loop / run_remove_zero_trip_loop**、**run DOMAIN=ast|pass**、**run DOMAIN=pass PASS=xxx**、**test**（全量）、**run_tests DOMAIN=ast|pass**、**test_ast / test_pass**。**不支持 make check**、**make pass**。

### 运行（run）

仅支持 **DOMAIN=ast** 与 **DOMAIN=pass**；不带参数时跑全部（ast + pass，有 LLVM 时含 pass）。

| 命令 | 说明 |
|------|------|
| `make run` | 运行全部（ast 可执行文件 + pass 目标，有 LLVM 时） |
| `make run DOMAIN=ast` | 只运行 ast（interpreter v1..v4、antlr、nfa_dfa） |
| `make run DOMAIN=pass` | 只运行 pass（所有 plugin 的 pass） |
| `make run DOMAIN=pass PASS=simple_pass` | 只运行 simple_pass 插件 |
| `make run DOMAIN=pass PASS=remove_trivial_block` | 只运行 remove_trivial_block |
| `make run DOMAIN=pass PASS=remove_trivial_loop` | 只运行 remove_trivial_loop |
| `make run DOMAIN=pass PASS=remove_zero_trip_loop` | 只运行 remove_zero_trip_loop |
| `make run_ast` | 等价于 `make run DOMAIN=ast` |
| `make run_pass` | 等价于 `make run DOMAIN=pass`（需 LLVM，跑全部 pass） |
| `make run_simple_pass` | 只跑 SimplePass 插件（my-peephole, my-cfg） |
| `make run_remove_trivial_block` | 只跑 RemoveTrivialBlockPass |
| `make run_remove_trivial_loop` | 只跑 RemoveTrivialLoopPass |
| `make run_remove_zero_trip_loop` | 只跑 RemoveZeroTripLoopPass |
| `cmake --build build --target run` | 从项目根目录运行全部 |
| `cmake --build build --target run -- DOMAIN=pass PASS=xxx` | 从项目根目录只运行指定 pass（**注意 `--` 后传 DOMAIN/PASS**） |
| `cmake --build build --target run_simple_pass` | 从项目根目录只运行 SimplePass |
| `cmake --build build --target run_remove_trivial_block` | 从项目根目录只运行 RemoveTrivialBlockPass |
| `cmake --build build --target run_remove_trivial_loop` | 从项目根目录只运行 RemoveTrivialLoopPass |
| `cmake --build build --target run_remove_zero_trip_loop` | 从项目根目录只运行 RemoveZeroTripLoopPass |

### 测试（test）

仅支持 **DOMAIN=ast** 与 **DOMAIN=pass**；不带参数时跑全部测试。**不支持 make check**。  
首次执行 **make run** 或 **make patch_test_domain** 后，`make test DOMAIN=ast|pass` 可生效（通过 `ctest_wrapper.sh` 修补 test 目标）。

| 命令 | 说明 |
|------|------|
| `make test` | 运行全部测试 |
| `make test DOMAIN=ast` | 只运行 ast 域测试（^V[1-4]_） |
| `make test DOMAIN=pass` | 只运行 pass 域测试（label pass） |
| `make run_tests DOMAIN=ast` | 只运行 ast 域测试（^V[1-4]_） |
| `make run_tests DOMAIN=pass` | 只运行 pass 域测试（label pass） |
| `make test_ast` | 同上，等价于 `make run_tests DOMAIN=ast` |
| `make test_pass` | 同上，等价于 `make run_tests DOMAIN=pass` |
| `cmake --build build --target test` | 从项目根目录运行全部测试 |
| `cmake --build build --target run_tests -- DOMAIN=pass` | 从项目根目录只运行 pass 测试 |
| `cmake --build build --target test_ast` / `test_pass` | 从项目根目录只运行对应域 |

pass 域 UT：  
- **Pass_SimplePass_Src**：pass 源码 C++ UT（直接跑 `SimplePeepholePass` / `RemoveEmptyBlockPass`，解析 IR 后断言优化结果）。  
- **Pass_SimplePass**：脚本测试（跑 opt + SimplePass 插件并校验输出 IR：无 `add i32 %x, 0`、有 `ret i32 %x`）。

### LLVM Pass 插件（opt 加载 .so）

在检测到 `llvm-config` 时，会构建 `src/pass/` 下的 **SimplePass** 等插件（`.so`），并通过 CMake target 封装「用 opt 加载插件并跑 pass」的命令。运行 pass 使用 **make run DOMAIN=pass** 或 **make run_pass**，**不支持 make pass**。

| 命令 | 说明 |
|------|------|
| `cmake --build build --target SimplePass` | 仅编译插件，生成 `build/src/pass/SimplePass.so` |
| `make run_simple_pass` | 用 opt 处理 `simple_pass.ll`，输出到 `simple_pass_opt.ll` |
| `make run_remove_trivial_block` | 用 opt 处理 `remove_trivial_block.ll` |
| `make run_remove_trivial_loop` | 用 opt 处理 `remove_trivial_loop.ll` |
| `make run_remove_zero_trip_loop` | 用 opt 处理 `remove_zero_trip_loop.ll` |

等价的手动命令（在项目根目录）：

```bash
opt -load-pass-plugin=build/src/pass/SimplePass.so -passes="my-peephole,my-cfg" src/pass/simple_pass.ll -S -o build/src/pass/simple_pass_opt.ll
```

推荐在 build 目录下使用：`make run DOMAIN=pass` 或 `make run_pass` 跑 pass。

### 小结

- **run**：`make run` 跑全部；`make run_ast` / `make run_pass` 只跑对应域；`make run DOMAIN=pass PASS=xxx` 或 `make run_xxx` 只跑单个 pass（`xxx` 可为 `simple_pass`、`remove_trivial_block`、`remove_trivial_loop`、`remove_zero_trip_loop`）。
- **test**：`make test` 跑全部，`make run_tests DOMAIN=ast|pass` 或 `make test_ast` / `make test_pass` 只跑对应域。
- 用 `cmake --build` 并要传 **DOMAIN** 时，写法为：`cmake --build build --target run -- DOMAIN=ast`（`--` 与参数之间用空格）。
