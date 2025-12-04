#include <iostream>

#include "interpreter_v3.hpp"

using namespace Interpreter_V3;

void runExample(const std::string &title, const std::string &code) {
  std::cout << "\n======= " << title << " =======\n";
  std::cout << "Code:\n" << code << "\n";
  std::cout << "Output:\n";

  try {
    auto env = std::make_shared<Env>();
    Parser p(code);
    auto stmts = p.parseProgram();
    for (auto &stmt : stmts)
      stmt->exec(env);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

int main() {
  std::cout << "===== Compiler V3 Demo: Advanced Features =====\n";

  // 1. Ternary Operator
  runExample("Ternary Operator", R"(
let x = 10;
let result = x > 5 ? "big" : "small";
print "x is", result;
)");

  // 2. Type System
  runExample("Type System (typeof)", R"(
let num = 42;
let pi = 3.14;
let text = "hello";
let flag = true;
print "Types:", typeof num, typeof pi, typeof text, typeof flag;
)");

  // 3. String Comparison (Bug Fix)
  runExample("String Comparison", R"(
let cmd = "start";
if (cmd == "start") {
    print "Command is start";
}
if (cmd == "stop") {
    print "Command is stop";
}
if (cmd != "stop") {
    print "Command is not stop";
}

fn processCommand(op) {
    if (op == "add") return "Adding...";
    if (op == "sub") return "Subtracting...";
    if (op == "mul") return "Multiplying...";
    return "Unknown command";
}
print processCommand("add");
print processCommand("mul");
print processCommand("div");
)");

  // 4. Compound Assignment
  runExample("Compound Assignment Operators", R"(
let x = 10;
print "Initial:", x;
x += 5;
print "After += 5:", x;
x *= 2;
print "After *= 2:", x;
x -= 10;
print "After -= 10:", x;
x /= 2;
print "After /= 2:", x;
)");

  // 5. Scope Chain
  runExample("Scope Chain & Shadowing", R"(
let x = "global";
print "Global x:", x;
{
    let x = "block";
    print "Block x:", x;
    {
        let x = "nested";
        print "Nested x:", x;
    }
    print "Back to block x:", x;
}
print "Back to global x:", x;
)");

  // 6. Closures
  runExample("Closures - Counter Factory", R"(
fn makeCounter() {
    let count = 0;
    fn increment() {
        count += 1;
        return count;
    }
    return increment;
}
let counter1 = makeCounter();
let counter2 = makeCounter();
print "Counter1:", counter1(), counter1(), counter1();
print "Counter2:", counter2(), counter2();
)");

  // 7. Higher-Order Functions
  runExample("Higher-Order Functions", R"(
fn apply(f, x) {
    return f(x);
}
fn double(n) {
    return n * 2;
}
fn square(n) {
    return n * n;
}
print "apply(double, 5) =", apply(double, 5);
print "apply(square, 5) =", apply(square, 5);
)");

  // 8. Recursive Closure
  runExample("Fibonacci with Recursion", R"(
fn fib(n) {
    if (n < 1) return 0;
    if (n == 1) return 1;
    return fib(n - 1) + fib(n - 2);
}
print "Fibonacci sequence:";
let i = 0;
while (i < 11) {
    print fib(i);
    i += 1;
}
)");

  // 9. Complex Expression with All Features
  runExample("Complex Example - All Features Combined", R"(
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
    
    fn reset() {
        value = initial;
        return value;
    }
    
    fn dispatcher(op, x) {
        if (op == "add") return add(x);
        if (op == "mul") return multiply(x);
        if (op == "get") return getValue();
        if (op == "reset") return reset();
        return 0;
    }
    
    return dispatcher;
}

let calc = makeCalculator(10);
print "Initial:", calc("get", 0);
print "After add 5:", calc("add", 5);
print "After mul 2:", calc("mul", 2);
print "Current:", calc("get", 0);
print "After reset:", calc("reset", 0);
)");

  return 0;
}
