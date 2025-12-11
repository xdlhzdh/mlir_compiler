#include <gtest/gtest.h>
#include <sstream>

#include "interpreter_v3.hpp"

using namespace Interpreter_V3;

// Helper to capture stdout
class CaptureStdout {
  std::ostringstream buffer;
  std::streambuf *old;

public:
  CaptureStdout() : old(std::cout.rdbuf(buffer.rdbuf())) {}
  ~CaptureStdout() { std::cout.rdbuf(old); }
  std::string get() { return buffer.str(); }
};

// ===== Lexer Tests =====
TEST(V3_LexerTest, TokenizeBasic) {
  Lexer lex("let x = 42;");
  EXPECT_EQ(lex.nextToken().type, TOK_LET);
  EXPECT_EQ(lex.nextToken().type, TOK_ID);
  EXPECT_EQ(lex.nextToken().type, TOK_ASSIGN);
  EXPECT_EQ(lex.nextToken().type, TOK_INT);
  EXPECT_EQ(lex.nextToken().type, TOK_SEMI);
}

TEST(V3_LexerTest, TokenizeDouble) {
  Lexer lex("3.14");
  Token tok = lex.nextToken();
  EXPECT_EQ(tok.type, TOK_DOUBLE);
  EXPECT_EQ(tok.text, "3.14");
}

TEST(V3_LexerTest, TokenizeTernary) {
  Lexer lex("x ? 1 : 0");
  lex.nextToken(); // x
  EXPECT_EQ(lex.nextToken().type, TOK_QUESTION);
  lex.nextToken(); // 1
  EXPECT_EQ(lex.nextToken().type, TOK_COLON);
}

TEST(V3_LexerTest, TokenizeCompoundAssign) {
  Lexer lex("x += 5;");
  lex.nextToken(); // x
  Token tok = lex.nextToken();
  EXPECT_EQ(tok.type, TOK_PLUS_ASSIGN);
  EXPECT_EQ(tok.text, "+=");
}

TEST(V3_LexerTest, TokenizeArrow) {
  Lexer lex("(x) => x * 2");
  lex.nextToken(); // (
  lex.nextToken(); // x
  lex.nextToken(); // )
  Token tok = lex.nextToken();
  EXPECT_EQ(tok.type, TOK_ARROW);
  EXPECT_EQ(tok.text, "=>");
}

TEST(V3_LexerTest, TokenizeBooleans) {
  Lexer lex("true false");
  EXPECT_EQ(lex.nextToken().type, TOK_TRUE);
  EXPECT_EQ(lex.nextToken().type, TOK_FALSE);
}

TEST(V3_LexerTest, TokenizeTypeof) {
  Lexer lex("typeof x");
  EXPECT_EQ(lex.nextToken().type, TOK_TYPEOF);
  EXPECT_EQ(lex.nextToken().type, TOK_ID);
}

// ===== Type System Tests =====
TEST(V3_TypeTest, IntType) {
  Value v(42);
  EXPECT_EQ(v.type, ValueType::INT);
  EXPECT_EQ(v.toInt(), 42);
  EXPECT_EQ(v.toNumber(), 42.0);
  EXPECT_TRUE(v.toBool());
  EXPECT_EQ(v.toString(), "42");
}

TEST(V3_TypeTest, DoubleType) {
  Value v(3.14);
  EXPECT_EQ(v.type, ValueType::DOUBLE);
  EXPECT_DOUBLE_EQ(v.toNumber(), 3.14);
  EXPECT_EQ(v.toInt(), 3);
  EXPECT_TRUE(v.toBool());
}

TEST(V3_TypeTest, StringType) {
  Value v("hello");
  EXPECT_EQ(v.type, ValueType::STRING);
  EXPECT_EQ(v.toString(), "hello");
  EXPECT_TRUE(v.toBool());
}

TEST(V3_TypeTest, BoolType) {
  Value v(true);
  EXPECT_EQ(v.type, ValueType::BOOL);
  EXPECT_TRUE(v.toBool());
  EXPECT_EQ(v.toInt(), 1);
  EXPECT_EQ(v.toString(), "true");
}

// ===== Expression Tests =====
TEST(V3_ExprTest, TernaryOperator) {
  auto env = std::make_shared<Env>();
  Parser p("let result = 10 > 5 ? 100 : 200; print result;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "100\n");
}

TEST(V3_ExprTest, NestedTernary) {
  auto env = std::make_shared<Env>();
  Parser p(
      "let x = 3; let result = x > 5 ? 10 : x > 2 ? 20 : 30; print result;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "20\n");
}

TEST(V3_ExprTest, TypeofOperator) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let x = 42;
    let y = "hello";
    let z = true;
    print typeof x, typeof y, typeof z;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "int string bool\n");
}

TEST(V3_ExprTest, DoubleArithmetic) {
  auto env = std::make_shared<Env>();
  Parser p("let x = 3.5 + 2.5; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "6.000000\n");
}

TEST(V3_ExprTest, ModuloOperator) {
  auto env = std::make_shared<Env>();
  Parser p("let x = 17 % 5; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "2\n");
}

TEST(V3_ExprTest, UnaryNegation) {
  auto env = std::make_shared<Env>();
  Parser p("let x = -10; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "-10\n");
}

TEST(V3_ExprTest, LogicalNot) {
  auto env = std::make_shared<Env>();
  Parser p("let x = !false; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "true\n");
}

// ===== Compound Assignment Tests =====
TEST(V3_AssignTest, PlusAssign) {
  auto env = std::make_shared<Env>();
  Parser p("let x = 10; x += 5; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "15\n");
}

TEST(V3_AssignTest, MinusAssign) {
  auto env = std::make_shared<Env>();
  Parser p("let x = 20; x -= 8; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "12\n");
}

TEST(V3_AssignTest, MulAssign) {
  auto env = std::make_shared<Env>();
  Parser p("let x = 3; x *= 4; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "12\n");
}

TEST(V3_AssignTest, DivAssign) {
  auto env = std::make_shared<Env>();
  Parser p("let x = 100; x /= 4; print x;");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "25\n");
}

TEST(V3_AssignTest, StringPlusAssign) {
  auto env = std::make_shared<Env>();
  Parser p(R"(let s = "Hello"; s += " World"; print s;)");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "Hello World\n");
}

// ===== Scope Chain Tests =====
TEST(V3_ScopeTest, NestedScopes) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let x = 10;
    {
      let y = 20;
      print x, y;
    }
    print x;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "10 20\n10\n");
}

TEST(V3_ScopeTest, ShadowingVariables) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let x = 100;
    {
      let x = 200;
      print x;
    }
    print x;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "200\n100\n");
}

TEST(V3_ScopeTest, ModifyOuterScope) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let x = 10;
    {
      x = 50;
    }
    print x;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "50\n");
}

// ===== Closure Tests =====
TEST(V3_ClosureTest, SimpleClosure) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn makeCounter() {
      let count = 0;
      fn counter() {
        count += 1;
        return count;
      }
      return counter;
    }
    let c = makeCounter();
    print c();
    print c();
    print c();
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "1\n2\n3\n");
}

TEST(V3_ClosureTest, MultipleClosures) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn makeAdder(x) {
      fn add(y) {
        return x + y;
      }
      return add;
    }
    let add5 = makeAdder(5);
    let add10 = makeAdder(10);
    print add5(3);
    print add10(3);
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "8\n13\n");
}

TEST(V3_ClosureTest, ClosureWithLoop) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn makeMultiplier(factor) {
      fn multiply(n) {
        return n * factor;
      }
      return multiply;
    }
    let double = makeMultiplier(2);
    let triple = makeMultiplier(3);
    let i = 1;
    while (i < 4) {
      print double(i), triple(i);
      i += 1;
    }
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "2 3\n4 6\n6 9\n");
}

// ===== Function Tests =====
TEST(V3_FunctionTest, ReturnFunction) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn identity(x) {
      return x;
    }
    fn getFunc() {
      return identity;
    }
    let f = getFunc();
    print f(42);
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "42\n");
}

TEST(V3_FunctionTest, HigherOrderFunction) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn apply(f, x) {
      return f(x);
    }
    fn square(n) {
      return n * n;
    }
    print apply(square, 5);
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "25\n");
}

// ===== Integration Tests =====
TEST(V3_IntegrationTest, ComplexExpression) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let a = 5;
    let b = 10;
    let result = a > 3 && b < 20 ? a * b : a + b;
    print result;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "50\n");
}

TEST(V3_IntegrationTest, FactorialWithClosure) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn factorial(n) {
      if (n < 1) return 1;
      return n * factorial(n - 1);
    }
    print factorial(5);
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "120\n");
}

TEST(V3_IntegrationTest, FibonacciWithMemoization) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn fib(n) {
      if (n < 1) return 0;
      if (n == 1) return 1;
      return fib(n - 1) + fib(n - 2);
    }
    print fib(10);
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "55\n");
}

TEST(V3_IntegrationTest, TypeCoercion) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let x = 10;
    let y = "20";
    print x + y;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "1020\n");
}

TEST(V3_IntegrationTest, MixedTypes) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let i = 42;
    let d = 3.14;
    let s = "test";
    let b = true;
    print typeof i, typeof d, typeof s, typeof b;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "int double string bool\n");
}

TEST(V3_IntegrationTest, ComplexControlFlow) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let sum = 0;
    let i = 1;
    while (i < 6) {
      if (i % 2 == 0) {
        sum += i;
      }
      i += 1;
    }
    print sum;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "6\n");
}

TEST(V3_IntegrationTest, NestedFunctionCalls) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn add(a, b) {
      return a + b;
    }
    fn multiply(a, b) {
      return a * b;
    }
    print multiply(add(2, 3), add(4, 5));
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "45\n");
}

// ===== String Comparison Tests =====
TEST(V3_StringComparisonTest, StringEquality) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let s1 = "hello";
    let s2 = "hello";
    let s3 = "world";
    print s1 == s2;
    print s1 == s3;
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "true\nfalse\n");
}

TEST(V3_StringComparisonTest, StringInequality) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let s1 = "hello";
    let s2 = "world";
    print s1 != s2;
    print s1 != "hello";
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "true\nfalse\n");
}

TEST(V3_StringComparisonTest, StringInIfCondition) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    let op = "add";
    if (op == "add") {
      print "Addition selected";
    }
    if (op == "sub") {
      print "Subtraction selected";
    }
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "Addition selected\n");
}

TEST(V3_StringComparisonTest, StringComparisonInFunction) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn getOperation(op) {
      if (op == "add") return 1;
      if (op == "sub") return 2;
      if (op == "mul") return 3;
      if (op == "div") return 4;
      return 0;
    }
    print getOperation("add");
    print getOperation("mul");
    print getOperation("unknown");
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "1\n3\n0\n");
}

TEST(V3_StringComparisonTest, MultipleStringComparisons) {
  auto env = std::make_shared<Env>();
  Parser p(R"(
    fn dispatcher(cmd, val) {
      if (cmd == "get") return val;
      if (cmd == "double") return val * 2;
      if (cmd == "square") return val * val;
      return 0;
    }
    print dispatcher("get", 10);
    print dispatcher("double", 10);
    print dispatcher("square", 10);
  )");
  auto stmts = p.parseProgram();

  CaptureStdout cap;
  for (auto &stmt : stmts)
    stmt->exec(env);
  EXPECT_EQ(cap.get(), "10\n20\n100\n");
}
