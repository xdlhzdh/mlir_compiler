# Interpreter V4 - Break and Continue Support

## Overview
Version 4 of the interpreter extends Version 3 by adding `break` and `continue` keywords for loop control flow in `while` loops.

## New Features

### 1. Break Statement
- **Syntax**: `break;`
- **Purpose**: Immediately exits the innermost `while` loop
- **Use Cases**:
  - Early termination when a condition is met
  - Search algorithms that stop when finding a result
  - Optimization to avoid unnecessary iterations

### 2. Continue Statement
- **Syntax**: `continue;`
- **Purpose**: Skips the rest of the current iteration and proceeds to the next iteration of the loop
- **Use Cases**:
  - Filtering items in a loop
  - Skipping invalid data
  - Implementing conditional processing

## Implementation Details

### Language Extensions
- Added `TOK_BREAK` and `TOK_CONTINUE` tokens to the lexer
- Implemented `BreakStmt` and `ContinueStmt` AST nodes
- Added `BreakException` and `ContinueException` for control flow
- Modified `WhileStmt` to catch and handle break/continue exceptions

### Exception-Based Control Flow
The implementation uses C++ exceptions to implement non-local control flow:
- `BreakException`: Thrown by `break` statement, caught by the innermost loop
- `ContinueException`: Thrown by `continue` statement, caught by the innermost loop
- This approach ensures break/continue only affect the innermost loop (not outer loops or functions)

## Code Examples

### Basic Break
```javascript
let i = 0;
while (i < 10) {
    if (i == 5) {
        break;  // Exit loop when i reaches 5
    }
    print i;
    i += 1;
}
// Output: 0, 1, 2, 3, 4
```

### Basic Continue
```javascript
let i = 0;
while (i < 10) {
    i += 1;
    if (i % 2 == 0) {
        continue;  // Skip even numbers
    }
    print i;
}
// Output: 1, 3, 5, 7, 9
```

### Nested Loops with Break
```javascript
let i = 0;
while (i < 3) {
    let j = 0;
    while (j < 5) {
        if (j == 2) {
            break;  // Only breaks inner loop
        }
        print j;
        j += 1;
    }
    i += 1;
}
// Inner loop breaks at j=2 for each outer iteration
```

### Combined Break and Continue
```javascript
let i = 0;
while (i < 20) {
    i += 1;
    if (i % 2 == 0) {
        continue;  // Skip even numbers
    }
    if (i > 10) {
        break;  // Stop when greater than 10
    }
    print i;
}
// Output: 1, 3, 5, 7, 9
```

## Files Added/Modified

### New Files
1. `include/interpreter_v4.hpp` - Complete interpreter implementation with break/continue
2. `src/demo_interpreter_v4.cpp` - Comprehensive demo with 12 examples
3. `tests/test_interpreter_v4.cpp` - 30 unit tests covering all features

### Modified Files
1. `CMakeLists.txt` - Added v4 build targets and test configuration

## Test Coverage

The test suite includes 30 tests organized in 7 categories:
1. **Lexer Tests** (3 tests) - Token recognition
2. **Break Tests** (4 tests) - Basic break functionality
3. **Continue Tests** (3 tests) - Basic continue functionality
4. **Break/Continue Combined** (8 tests) - Complex scenarios
5. **Integration Tests** (4 tests) - Integration with V3 features
6. **Parser Tests** (6 tests) - Syntax validation
7. **Comprehensive Tests** (2 tests) - Real-world patterns

All tests pass successfully (100% pass rate).

## Build Instructions

```bash
cd build
cmake ..
make demo_interpreter_v4 test_interpreter_v4
```

## Run Demo
```bash
./build/demo_interpreter_v4
```

## Run Tests
```bash
./build/test_interpreter_v4
# or
cd build && ctest -R V4
```

## Compatibility

V4 maintains full backward compatibility with V3 features:
- Ternary operator (`? :`)
- Type system (`typeof`)
- String comparison
- Compound assignment operators (`+=`, `-=`, `*=`, `/=`)
- Closures and higher-order functions
- Scope chain

## Future Enhancements

Potential improvements for V5:
- `for` loop support with break/continue
- `do-while` loop
- Labeled break/continue for breaking outer loops
- Switch/case statements
