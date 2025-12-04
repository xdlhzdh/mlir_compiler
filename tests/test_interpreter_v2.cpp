#include <functional>
#include <gtest/gtest.h>
#include <sstream>

#include "interpreter_v2.hpp"

namespace Interpreter_V2 {

// Helper function to capture stdout
std::string captureOutput(std::function<void()> func) {
  std::stringstream buffer;
  std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());
  func();
  std::cout.rdbuf(old);
  return buffer.str();
}

// Test Lexer tokenization
TEST(V2_LexerTest, TokenizeNumbers) {
  src = "123 456";
  pos = 0;
  Token t1 = nextToken();
  EXPECT_EQ(t1.type, TOK_NUM);
  EXPECT_EQ(t1.text, "123");

  Token t2 = nextToken();
  EXPECT_EQ(t2.type, TOK_NUM);
  EXPECT_EQ(t2.text, "456");
}

TEST(V2_LexerTest, TokenizeIdentifiers) {
  src = "abc xyz";
  pos = 0;
  Token t1 = nextToken();
  EXPECT_EQ(t1.type, TOK_ID);
  EXPECT_EQ(t1.text, "abc");

  Token t2 = nextToken();
  EXPECT_EQ(t2.type, TOK_ID);
  EXPECT_EQ(t2.text, "xyz");
}

TEST(V2_LexerTest, TokenizeKeywords) {
  src = "let fn return if else while print";
  pos = 0;
  EXPECT_EQ(nextToken().type, TOK_LET);
  EXPECT_EQ(nextToken().type, TOK_FN);
  EXPECT_EQ(nextToken().type, TOK_RETURN);
  EXPECT_EQ(nextToken().type, TOK_IF);
  EXPECT_EQ(nextToken().type, TOK_ELSE);
  EXPECT_EQ(nextToken().type, TOK_WHILE);
  EXPECT_EQ(nextToken().type, TOK_PRINT);
}

TEST(V2_LexerTest, TokenizeStrings) {
  src = R"("hello" "world")";
  pos = 0;
  Token t1 = nextToken();
  EXPECT_EQ(t1.type, TOK_STR);
  EXPECT_EQ(t1.text, "hello");

  Token t2 = nextToken();
  EXPECT_EQ(t2.type, TOK_STR);
  EXPECT_EQ(t2.text, "world");
}

TEST(V2_LexerTest, TokenizeOperators) {
  src = "+ - * / == != < > <= >= && ||";
  pos = 0;
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_EQ);
  EXPECT_EQ(nextToken().type, TOK_NEQ);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
  EXPECT_EQ(nextToken().type, TOK_OP);
}

// Test Value and Expression evaluation
TEST(V2_ExprTest, NumberExpression) {
  Env env;
  NumExpr expr(42);
  Value v = expr.eval(env);
  EXPECT_FALSE(v.isStr);
  EXPECT_EQ(v.i, 42);
}

TEST(V2_ExprTest, StringExpression) {
  Env env;
  StrExpr expr("hello");
  Value v = expr.eval(env);
  EXPECT_TRUE(v.isStr);
  EXPECT_EQ(v.s, "hello");
}

TEST(V2_ExprTest, VariableExpression) {
  Env env;
  env.setLocal("x", Value(100));
  VarExpr expr("x");
  Value v = expr.eval(env);
  EXPECT_EQ(v.i, 100);
}

TEST(V2_ExprTest, BinaryArithmetic) {
  Env env;

  // 5 + 3 = 8
  BinaryExpr add("+", new NumExpr(5), new NumExpr(3));
  EXPECT_EQ(add.eval(env).i, 8);

  // 10 - 4 = 6
  BinaryExpr sub("-", new NumExpr(10), new NumExpr(4));
  EXPECT_EQ(sub.eval(env).i, 6);

  // 6 * 7 = 42
  BinaryExpr mul("*", new NumExpr(6), new NumExpr(7));
  EXPECT_EQ(mul.eval(env).i, 42);

  // 20 / 4 = 5
  BinaryExpr div("/", new NumExpr(20), new NumExpr(4));
  EXPECT_EQ(div.eval(env).i, 5);
}

TEST(V2_ExprTest, BinaryComparison) {
  Env env;

  // 5 == 5 -> true (1)
  BinaryExpr eq("==", new NumExpr(5), new NumExpr(5));
  EXPECT_EQ(eq.eval(env).i, 1);

  // 5 != 3 -> true (1)
  BinaryExpr neq("!=", new NumExpr(5), new NumExpr(3));
  EXPECT_EQ(neq.eval(env).i, 1);

  // 3 < 5 -> true (1)
  BinaryExpr lt("<", new NumExpr(3), new NumExpr(5));
  EXPECT_EQ(lt.eval(env).i, 1);

  // 5 > 3 -> true (1)
  BinaryExpr gt(">", new NumExpr(5), new NumExpr(3));
  EXPECT_EQ(gt.eval(env).i, 1);
}

TEST(V2_ExprTest, BinaryLogical) {
  Env env;

  // 1 && 1 -> true (1)
  BinaryExpr and_expr("&&", new NumExpr(1), new NumExpr(1));
  EXPECT_EQ(and_expr.eval(env).i, 1);

  // 0 || 1 -> true (1)
  BinaryExpr or_expr("||", new NumExpr(0), new NumExpr(1));
  EXPECT_EQ(or_expr.eval(env).i, 1);
}

TEST(V2_ExprTest, StringConcatenation) {
  Env env;

  // "hello" + " world"
  BinaryExpr concat("+", new StrExpr("hello"), new StrExpr(" world"));
  Value v = concat.eval(env);
  EXPECT_TRUE(v.isStr);
  EXPECT_EQ(v.s, "hello world");
}

// Test Statements
TEST(V2_StmtTest, LetStatement) {
  Env env;
  LetStmt stmt("x", new NumExpr(42));
  stmt.exec(env);
  EXPECT_EQ(env.get("x").i, 42);
}

TEST(V2_StmtTest, AssignStatement) {
  Env env;
  env.setLocal("x", Value(10));
  AssignStmt stmt("x", new NumExpr(20));
  stmt.exec(env);
  EXPECT_EQ(env.get("x").i, 20);
}

TEST(V2_StmtTest, PrintStatement) {
  Env env;
  std::vector<Expr *> exprs;
  exprs.push_back(new NumExpr(42));
  exprs.push_back(new StrExpr("test"));
  PrintStmt stmt(exprs);

  std::string output = captureOutput([&]() { stmt.exec(env); });
  EXPECT_EQ(output, "42 test\n");
}

TEST(V2_StmtTest, IfStatement) {
  Env env;

  // if (1) { x = 10; } else { x = 20; }
  IfStmt stmt(new NumExpr(1), new LetStmt("x", new NumExpr(10)),
              new LetStmt("x", new NumExpr(20)));
  stmt.exec(env);
  EXPECT_EQ(env.get("x").i, 10);
}

TEST(V2_StmtTest, WhileStatement) {
  Env env;
  env.setLocal("i", Value(3));

  // while (i > 0) { i = i - 1; }
  BlockStmt *body = new BlockStmt();
  body->body.push_back(new AssignStmt(
      "i", new BinaryExpr("-", new VarExpr("i"), new NumExpr(1))));
  WhileStmt stmt(new BinaryExpr(">", new VarExpr("i"), new NumExpr(0)), body);

  stmt.exec(env);
  EXPECT_EQ(env.get("i").i, 0);
}

// Test Parser
TEST(V2_ParserTest, ParseNumber) {
  src = "42";
  pos = 0;
  getNext();
  Expr *expr = parseExpression();
  Env env;
  EXPECT_EQ(expr->eval(env).i, 42);
}

TEST(V2_ParserTest, ParseString) {
  src = R"("hello")";
  pos = 0;
  getNext();
  Expr *expr = parseExpression();
  Env env;
  Value v = expr->eval(env);
  EXPECT_TRUE(v.isStr);
  EXPECT_EQ(v.s, "hello");
}

TEST(V2_ParserTest, ParseArithmetic) {
  src = "1 + 2 * 3";
  pos = 0;
  getNext();
  Expr *expr = parseExpression();
  Env env;
  EXPECT_EQ(expr->eval(env).i, 7); // respects precedence
}

TEST(V2_ParserTest, ParseComparison) {
  src = "5 > 3";
  pos = 0;
  getNext();
  Expr *expr = parseExpression();
  Env env;
  EXPECT_EQ(expr->eval(env).i, 1);
}

TEST(V2_ParserTest, ParseLetStatement) {
  src = "let x = 42;";
  pos = 0;
  getNext();
  Stmt *stmt = parseStatement();
  Env env;
  stmt->exec(env);
  EXPECT_EQ(env.get("x").i, 42);
}

TEST(V2_ParserTest, ParseAssignStatement) {
  src = "x = 100;";
  pos = 0;
  getNext();
  Env env;
  env.setLocal("x", Value(0));
  Stmt *stmt = parseStatement();
  stmt->exec(env);
  EXPECT_EQ(env.get("x").i, 100);
}

TEST(V2_ParserTest, ParsePrintStatement) {
  src = R"(print("hello", 42);)";
  pos = 0;
  getNext();
  Stmt *stmt = parseStatement();
  Env env;

  std::string output = captureOutput([&]() { stmt->exec(env); });
  EXPECT_EQ(output, "hello 42\n");
}

TEST(V2_ParserTest, ParseIfStatement) {
  src = "let x = 0; if (1) { x = 10; }";
  pos = 0;
  getNext();
  Env env;

  // Parse let statement first
  Stmt *let_stmt = parseStatement();
  let_stmt->exec(env);

  // Parse if statement
  Stmt *if_stmt = parseStatement();
  if_stmt->exec(env);

  EXPECT_EQ(env.get("x").i, 10);
}

TEST(V2_ParserTest, ParseWhileStatement) {
  src = "let i = 3; while (i > 0) { i = i - 1; }";
  pos = 0;
  getNext();
  Env env;

  Stmt *let_stmt = parseStatement();
  let_stmt->exec(env);

  Stmt *while_stmt = parseStatement();
  while_stmt->exec(env);

  EXPECT_EQ(env.get("i").i, 0);
}

// Integration tests
TEST(V2_IntegrationTest, SimpleProgram) {
  std::string code = R"(
    let x = 5;
    let y = 10;
    let z = x + y;
  )";

  src = code;
  pos = 0;
  getNext();

  std::vector<Stmt *> program;
  while (curTok.type != TOK_EOF) {
    program.push_back(parseStatement());
  }

  Env env;
  for (auto stmt : program) {
    stmt->exec(env);
  }

  EXPECT_EQ(env.get("x").i, 5);
  EXPECT_EQ(env.get("y").i, 10);
  EXPECT_EQ(env.get("z").i, 15);
}

TEST(V2_IntegrationTest, FunctionCall) {
  std::string code = R"(
    fn add(a, b) {
      return a + b;
    }
    let result = add(3, 4);
  )";

  src = code;
  pos = 0;
  funcTable.clear();
  getNext();

  // Parse function
  parseFunction();

  // Parse let statement
  Stmt *stmt = parseStatement();

  Env env;
  stmt->exec(env);

  EXPECT_EQ(env.get("result").i, 7);
}

TEST(V2_IntegrationTest, Factorial) {
  std::string code = R"(
    fn fact(n) {
      if (n == 0) { return 1; }
      return n * fact(n - 1);
    }
    let result = fact(5);
  )";

  src = code;
  pos = 0;
  funcTable.clear();
  getNext();

  parseFunction();
  Stmt *stmt = parseStatement();

  Env env;
  stmt->exec(env);

  EXPECT_EQ(env.get("result").i, 120);
}

TEST(V2_IntegrationTest, StringOperations) {
  std::string code = R"(
    let s1 = "Hello";
    let s2 = " World";
    let s3 = s1 + s2;
  )";

  src = code;
  pos = 0;
  getNext();

  std::vector<Stmt *> program;
  while (curTok.type != TOK_EOF) {
    program.push_back(parseStatement());
  }

  Env env;
  for (auto stmt : program) {
    stmt->exec(env);
  }

  Value v = env.get("s3");
  EXPECT_TRUE(v.isStr);
  EXPECT_EQ(v.s, "Hello World");
}

TEST(V2_IntegrationTest, LoopWithCondition) {
  std::string code = R"(
    let sum = 0;
    let i = 1;
    while (i <= 5) {
      sum = sum + i;
      i = i + 1;
    }
  )";

  src = code;
  pos = 0;
  getNext();

  std::vector<Stmt *> program;
  while (curTok.type != TOK_EOF) {
    program.push_back(parseStatement());
  }

  Env env;
  for (auto stmt : program) {
    stmt->exec(env);
  }

  EXPECT_EQ(env.get("sum").i, 15); // 1+2+3+4+5 = 15
  EXPECT_EQ(env.get("i").i, 6);
}

} // namespace Interpreter_V2
