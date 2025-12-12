# Interpreter V3 - Advanced Features Description

## Overview
Interpreter V3 builds upon V2 by adding the following advanced features, making it a more complete programming language compiler.

## New Features

### 1. Ternary Operator
Supports conditional expressions `condition ? true_expr : false_expr`

```javascript
let x = 10;
let result = x > 5 ? "big" : "small";  // result = "big"
let nested = x > 10 ? "a" : x > 5 ? "b" : "c";  // Supports nesting
```

### 2. Type System
- Supports multiple data types: `INT`, `DOUBLE`, `STRING`, `BOOL`, `FUNCTION`
- `typeof` operator can query the type of a value
- Automatic type conversion and explicit casting

```javascript
let num = 42;           // int
let pi = 3.14;          // double
let text = "hello";     // string
let flag = true;        // bool

print typeof num;       // "int"
print typeof pi;        // "double"
print typeof text;      // "string"
```

### 3. Compound Assignment Operators
Supports `+=`, `-=`, `*=`, `/=` operators

```javascript
let x = 10;
x += 5;   // x = 15
x *= 2;   // x = 30
x -= 10;  // x = 20
x /= 2;   // x = 10
```

String concatenation:
```javascript
let s = "Hello";
s += " World";  // s = "Hello World"
```

### 4. Scope Chain
- Scope chain based on parent-child relationships
- Supports variable shadowing
- Inner scopes can access and modify outer variables

```javascript
let x = "global";
{
    let x = "block";  // Shadows the outer x
    print x;          // "block"
}
print x;              // "global"
```

### 5. Closures
Functions can capture the environment at the time of their definition, enabling true closures

```javascript
fn makeCounter() {
    let count = 0;
    fn increment() {
        count += 1;  // Captures the outer count
        return count;
    }
    return increment;
}

let counter1 = makeCounter();
print counter1();  // 1
print counter1();  // 2
print counter1();  // 3

let counter2 = makeCounter();  // Independent counter
print counter2();  // 1
```

### 6. Higher-Order Functions
Functions can be passed as arguments and returned as values

```javascript
fn apply(f, x) {
    return f(x);
}

fn double(n) {
    return n * 2;
}

print apply(double, 5);  // 10
```

Factory pattern with closures:
```javascript
fn makeAdder(x) {
    fn add(y) {
        return x + y;
    }
    return add;
}

let add5 = makeAdder(5);
let add10 = makeAdder(10);
print add5(3);   // 8
print add10(3);  // 13
```

### 7. More Robust Expression Parsing
- Correct operator precedence
- Supports unary operators: `-` (negation), `!` (logical NOT)
- Supports comparison operators: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Supports logical operators: `&&`, `||`
- Supports arithmetic operators: `+`, `-`, `*`, `/`, `%` (modulo)

Precedence order (from highest to lowest):
1. Unary operators: `-`, `!`, `typeof`
2. Multiplication, division, modulo: `*`, `/`, `%`
3. Addition, subtraction: `+`, `-`
4. Comparison: `<`, `>`, `<=`, `>=`
5. Equality: `==`, `!=`
6. Logical AND: `&&`
7. Logical OR: `||`
8. Ternary: `? :`

### 8. Other Improvements
- **Double Type**: Supports floating-point numbers like `3.14`
- **Boolean Literals**: `true` and `false`
- **Modulo Operator**: `%` for modulo operations
- **Better Error Messages**: Provides clear parsing error messages

## Comparison with V2

| Feature | V2 | V3 |
|---------|----|----|
| Data Types | Integers, Strings | Integers, Floats, Strings, Booleans |
| Ternary Operator | ❌ | ✅ |
| typeof Operator | ❌ | ✅ |
| Compound Assignment | ❌ | ✅ |
| Scope Chain | Simple Stack | Parent-Child Links |
| Closures | Partial Support | Full Support |
| Higher-Order Functions | Basic Support | Full Support |
| Modulo Operator | ❌ | ✅ |
| Unary Operators | ❌ | ✅ |

## Test Results
- **V1 Tests**: All 22 tests passed ✅
- **V2 Tests**: All 31 tests passed ✅
- **V3 Tests**: All 38 tests passed ✅

## Compilation and Execution

### Compile All Tests
```bash
cd build
cmake ..
make
```

### Run Tests
```bash
./compiler_v1_test  # Run V1 tests
./compiler_v2_test  # Run V2 tests
./compiler_v3_test  # Run V3 tests
```

### Run Demo
```bash
g++ -std=c++17 -I.. ../demo_v3.cpp -o demo_v3
./demo_v3
```

## Code Examples

### Complete Example: Calculator Closure
```javascript
fn makeCalculator(initial) {
    let value = initial;
    
    fn add(x) {
        value += x;
        return value;
    }
    
    fn multiply(x) {
        value *= x;
        return value;
    }
    
    fn getValue() {
        return value;
    }
    
    return add;  // Return closure
}

let calc = makeCalculator(10);
print calc(5);   // 15
print calc(3);   // 18
```

### Recursion with Closures
```javascript
fn factorial(n) {
    if (n < 1) return 1;
    return n * factorial(n - 1);
}

fn fibonacci(n) {
    if (n < 1) return 0;
    if (n == 1) return 1;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

print factorial(5);   // 120
print fibonacci(10);  // 55
```

## Technical Implementation Details

### Memory Management
- Uses `std::shared_ptr` to manage function bodies, allowing multiple closures to share the same code
- Uses `std::enable_shared_from_this` to implement self-referencing environments
- Uses `std::variant` to implement polymorphic value types

### Scope Implementation
- Each `Env` object holds a `shared_ptr` to its parent environment
- Variable lookup searches up the scope chain
- New scopes are created in `BlockStmt::exec()`

### Closure Implementation
- Function objects capture the environment at the time of their definition (`capturedEnv`)
- A new environment is created during invocation, with the parent environment set to the captured environment
- Parameters are defined in the new environment

## Potential Future Extensions
- Lambda expressions: `(x) => x * 2`
- Array/List support
- Object/Map types
- Classes and inheritance
- Module system
- Exception handling
- Iterators and generators

## Summary
Version 3 implements a highly functional dynamic programming language, supporting closures, higher-order functions, scope chains, and other advanced features. It serves as an excellent example for learning compiler principles and functional programming concepts.
