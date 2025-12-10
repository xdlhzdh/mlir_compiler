#include "InterpreterVisitor.hpp"
#include "generated/LangLexer.h"
#include "generated/LangParser.h"
#include <antlr4-runtime.h>
#include <iostream>

using namespace antlr4;
using namespace LangInterpreter;

void runCode(const std::string &code, const std::string &testName = "") {
  if (!testName.empty()) {
    std::cout << "=== " << testName << " ===\n";
  } else {
    std::cout << "Running code:\n" << code << "\n";
  }
  std::cout << "Output:\n";

  try {
    ANTLRInputStream input(code);
    LangLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    LangParser parser(&tokens);

    auto tree = parser.program();
    InterpreterVisitor visitor;
    visitor.visitProgram(tree);

    if (!testName.empty()) {
      std::cout << "✓ Success\n";
    }
    std::cout << std::endl;
  } catch (const std::exception &e) {
    std::cerr << (testName.empty() ? "Error: " : "✗ Error: ") << e.what()
              << std::endl
              << std::endl;
  }
}

int main() {
  std::cout << "=== ANTLR-based Interpreter Demo ===\n\n";

  // Test 1: Simple print
  runCode(R"(
print(42);
)",
          "Test 1: Simple print");

  // Test 2: Basic arithmetic
  runCode(R"(
let x = 10;
let y = 20;
print(x + y);
)",
          "Test 2: Basic arithmetic");

  // Test 3: String concatenation
  runCode(R"(
let name = "Alice";
let greeting = "Hello, " + name;
print(greeting);
)",
          "Test 3: String concatenation");

  // Test 4: Functions
  runCode(R"(
fn add(a, b) {
  return a + b;
}
let result = add(5, 3);
print(result);
)",
          "Test 4: Functions");

  // Test 5: Closures - basic counter
  runCode(R"(
fn makeCounter() {
  let count = 0;
  fn increment() {
    count = count + 1;
    return count;
  }
  return increment;
}
let counter = makeCounter();
print(counter());
print(counter());
print(counter());
)",
          "Test 5: Basic closure - counter");

  // Test 6: Closures - modifying captured variables
  runCode(R"(
fn makeCounter() {
  let count = 0;
  fn increment() {
    count = count + 1;
    print(count);
  }
  return increment;
}
let counter = makeCounter();
counter();
counter();
counter();
)",
          "Test 6: Modifying captured variables");

  // Test 7: Multiple closures with independent state
  runCode(R"(
fn makeCounter() {
  let count = 0;
  fn increment() {
    count = count + 1;
    return count;
  }
  return increment;
}
let c1 = makeCounter();
let c2 = makeCounter();
print(c1());
print(c1());
print(c2());
print(c1());
)",
          "Test 7: Independent closure state");

  // Test 8: Nested closures
  runCode(R"(
fn outer(x) {
  fn middle(y) {
    fn inner(z) {
      print(x + y + z);
    }
    return inner;
  }
  return middle;
}
let f = outer(1);
let g = f(2);
g(3);
)",
          "Test 8: Nested closures");

  // Test 9: Closure capturing multiple variables
  runCode(R"(
fn makeAdder(a, b) {
  fn add(c) {
    print(a + b + c);
  }
  return add;
}
let add10 = makeAdder(5, 5);
add10(3);
add10(7);
)",
          "Test 9: Multiple variable capture");

  // Test 10: Closures created in sequence
  runCode(R"(
fn makeMultiplier(factor) {
  fn multiply(x) {
    return factor * x;
  }
  return multiply;
}
let double = makeMultiplier(2);
let triple = makeMultiplier(3);
print(double(5));
print(triple(5));
)",
          "Test 10: Sequential closure creation");

  // Test 11: Returning getter/setter functions
  runCode(R"(
fn makeGetterSetter() {
  let value = 0;
  fn get() {
    return value;
  }
  fn set(v) {
    value = v;
  }
  return get;
}
let getter = makeGetterSetter();
print(getter());
)",
          "Test 11: Getter/setter pattern");

  // Test 12: While loop with break
  runCode(R"(
let i = 0;
while (i < 10) {
  if (i == 5) {
    break;
  }
  print(i);
  i = i + 1;
}
)",
          "Test 12: While loop with break");

  // Test 13: While loop with continue
  runCode(R"(
let i = 0;
while (i < 5) {
  i = i + 1;
  if (i == 3) {
    continue;
  }
  print(i);
}
)",
          "Test 13: While loop with continue");

  // Test 14: Ternary operator
  runCode(R"(
let x = 10;
let result = x > 5 ? "big" : "small";
print(result);
)",
          "Test 14: Ternary operator");

  // Test 15: Compound assignment
  runCode(R"(
let x = 10;
x += 5;
print(x);
x *= 2;
print(x);
)",
          "Test 15: Compound assignment");

  // Test 16: Typeof operator
  runCode(R"(
let x = 42;
let y = 3.14;
let z = "hello";
print(typeof x);
print(typeof y);
print(typeof z);
)",
          "Test 16: Typeof operator");

  // Test 17: Nested loops with break/continue
  runCode(R"(
let i = 0;
while (i < 3) {
  let j = 0;
  while (j < 3) {
    if (j == 1) {
      j = j + 1;
      continue;
    }
    print(i * 10 + j);
    j = j + 1;
  }
  i = i + 1;
}
)",
          "Test 17: Nested loops with break/continue");

  // Test 18: Integer and double mixed arithmetic
  runCode(R"(
let a = 10;
let b = 3.5;
print(a + b);
print(a - b);
print(a * b);
print(a / b);
let c = 2.5;
let d = 1.5;
print(c + d);
print(c * d);
)",
          "Test 18: Integer and double mixed arithmetic");

  // Test 19: Prefix decrement operator
  runCode(R"(
let x = 10;
print(--x);
print(x);
print(--x);
print(x);
let y = 5;
while (y > 0) {
  print(--y);
}
)",
          "Test 19: Prefix decrement operator");

  // Test 20: Decrement in expressions
  runCode(R"(
let i = 10;
let j = --i + 5;
print(i);
print(j);
let k = 3;
print(--k * 2);
print(k);
)",
          "Test 20: Decrement in expressions");

  // Test 21: Postfix decrement operator (i--)
  runCode(R"(
let x = 10;
print(x--);
print(x);
print(x--);
print(x);
let y = 5;
while (y > 0) {
  print(y--);
}
)",
          "Test 21: Postfix decrement operator");

  // Test 22: Prefix vs Postfix decrement
  runCode(R"(
let a = 10;
let b = 10;
print(--a);
print(a);
print(b--);
print(b);
let i = 5;
let j = --i + i--;
print(i);
print(j);
)",
          "Test 22: Prefix vs Postfix decrement");

  // Test 23: Anonymous functions
  runCode(R"(
let add = fn(a, b) => {
  return a + b;
};
print(add(3, 4));
print(add(10, 20));
)",
          "Test 23: Anonymous functions");

  // Test 24: Anonymous function as argument
  runCode(R"(
fn apply(f, x, y) {
  return f(x, y);
}
let result = apply(fn(a, b) => {
  return a * b;
}, 5, 6);
print(result);
)",
          "Test 24: Anonymous function as argument");

  // Test 25: Anonymous function with closure
  runCode(R"(
fn makeMultiplier(factor) {
  return fn(x) => {
    return factor * x;
  };
}
let times3 = makeMultiplier(3);
let times5 = makeMultiplier(5);
print(times3(4));
print(times5(4));
)",
          "Test 25: Anonymous function with closure");

  std::cout << "=== All tests completed ===\n";
  return 0;
}
