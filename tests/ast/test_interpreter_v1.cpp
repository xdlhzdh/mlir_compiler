#include <gtest/gtest.h>
#include <map>
#include <sstream>

#include "interpreter_v1.hpp"

namespace Interpreter_V1 {
// Test Lexer tokenization
TEST(V1_LexerTest, TokenizeNumbers) {
  Lexer lex("123 456");
  Token t1 = lex.nextToken();
  EXPECT_EQ(t1.type, tok_number);
  EXPECT_EQ(t1.number, 123);

  Token t2 = lex.nextToken();
  EXPECT_EQ(t2.type, tok_number);
  EXPECT_EQ(t2.number, 456);
}

TEST(V1_LexerTest, TokenizeIdentifiers) {
  Lexer lex("abc xyz");
  Token t1 = lex.nextToken();
  EXPECT_EQ(t1.type, tok_identifier);
  EXPECT_EQ(t1.text, "abc");

  Token t2 = lex.nextToken();
  EXPECT_EQ(t2.type, tok_identifier);
  EXPECT_EQ(t2.text, "xyz");
}

TEST(V1_LexerTest, TokenizePrintKeyword) {
  Lexer lex("print");
  Token t = lex.nextToken();
  EXPECT_EQ(t.type, tok_print);
}

TEST(V1_LexerTest, TokenizeOperators) {
  Lexer lex("+ - * / = ( ) ;");
  EXPECT_EQ(lex.nextToken().type, tok_plus);
  EXPECT_EQ(lex.nextToken().type, tok_minus);
  EXPECT_EQ(lex.nextToken().type, tok_mul);
  EXPECT_EQ(lex.nextToken().type, tok_div);
  EXPECT_EQ(lex.nextToken().type, tok_equal);
  EXPECT_EQ(lex.nextToken().type, tok_lparen);
  EXPECT_EQ(lex.nextToken().type, tok_rparen);
  EXPECT_EQ(lex.nextToken().type, tok_semicolon);
}

// Test AST evaluation
TEST(V1_ASTTest, NumberExpression) {
  std::map<std::string, int> ctx;
  NumberExprAST expr(42);
  EXPECT_EQ(expr.eval(ctx), 42);
}

TEST(V1_ASTTest, VariableExpression) {
  std::map<std::string, int> ctx;
  ctx["x"] = 10;
  VariableExprAST expr("x");
  EXPECT_EQ(expr.eval(ctx), 10);
}

TEST(V1_ASTTest, BinaryAddition) {
  std::map<std::string, int> ctx;
  auto lhs = std::make_unique<NumberExprAST>(5);
  auto rhs = std::make_unique<NumberExprAST>(3);
  BinaryExprAST expr('+', std::move(lhs), std::move(rhs));
  EXPECT_EQ(expr.eval(ctx), 8);
}

TEST(V1_ASTTest, BinarySubtraction) {
  std::map<std::string, int> ctx;
  auto lhs = std::make_unique<NumberExprAST>(10);
  auto rhs = std::make_unique<NumberExprAST>(4);
  BinaryExprAST expr('-', std::move(lhs), std::move(rhs));
  EXPECT_EQ(expr.eval(ctx), 6);
}

TEST(V1_ASTTest, BinaryMultiplication) {
  std::map<std::string, int> ctx;
  auto lhs = std::make_unique<NumberExprAST>(6);
  auto rhs = std::make_unique<NumberExprAST>(7);
  BinaryExprAST expr('*', std::move(lhs), std::move(rhs));
  EXPECT_EQ(expr.eval(ctx), 42);
}

TEST(V1_ASTTest, BinaryDivision) {
  std::map<std::string, int> ctx;
  auto lhs = std::make_unique<NumberExprAST>(20);
  auto rhs = std::make_unique<NumberExprAST>(4);
  BinaryExprAST expr('/', std::move(lhs), std::move(rhs));
  EXPECT_EQ(expr.eval(ctx), 5);
}

TEST(V1_ASTTest, ComplexExpression) {
  std::map<std::string, int> ctx;
  // (3 + 5) * 2 = 16
  auto lhs =
      std::make_unique<BinaryExprAST>('+', std::make_unique<NumberExprAST>(3),
                                      std::make_unique<NumberExprAST>(5));
  auto expr =
      BinaryExprAST('*', std::move(lhs), std::make_unique<NumberExprAST>(2));
  EXPECT_EQ(expr.eval(ctx), 16);
}

// Test Parser
TEST(V1_ParserTest, ParseNumber) {
  Lexer lex("42");
  Parser parser(lex);
  auto expr = parser.parseExpr();
  std::map<std::string, int> ctx;
  EXPECT_EQ(expr->eval(ctx), 42);
}

TEST(V1_ParserTest, ParseAddition) {
  Lexer lex("1 + 2");
  Parser parser(lex);
  auto expr = parser.parseExpr();
  std::map<std::string, int> ctx;
  EXPECT_EQ(expr->eval(ctx), 3);
}

TEST(V1_ParserTest, ParseMultiplication) {
  Lexer lex("3 * 4");
  Parser parser(lex);
  auto expr = parser.parseExpr();
  std::map<std::string, int> ctx;
  EXPECT_EQ(expr->eval(ctx), 12);
}

TEST(V1_ParserTest, ParseComplexExpression) {
  Lexer lex("1 + 2 * 3");
  Parser parser(lex);
  auto expr = parser.parseExpr();
  std::map<std::string, int> ctx;
  EXPECT_EQ(expr->eval(ctx), 7); // respects precedence
}

TEST(V1_ParserTest, ParseParentheses) {
  Lexer lex("(1 + 2) * 3");
  Parser parser(lex);
  auto expr = parser.parseExpr();
  std::map<std::string, int> ctx;
  EXPECT_EQ(expr->eval(ctx), 9);
}

TEST(V1_ParserTest, ParseVariable) {
  Lexer lex("x");
  Parser parser(lex);
  auto expr = parser.parseExpr();
  std::map<std::string, int> ctx;
  ctx["x"] = 100;
  EXPECT_EQ(expr->eval(ctx), 100);
}

TEST(V1_ParserTest, ParseAssignmentStatement) {
  Lexer lex("x = 42;");
  Parser parser(lex);
  auto stmt = parser.parseStmt();
  std::map<std::string, int> ctx;
  stmt->exec(ctx);
  EXPECT_EQ(ctx["x"], 42);
}

TEST(V1_ParserTest, ParsePrintStatement) {
  Lexer lex("print(5);");
  Parser parser(lex);
  auto stmt = parser.parseStmt();
  std::map<std::string, int> ctx;

  // Redirect stdout to capture print output
  std::stringstream buffer;
  std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

  stmt->exec(ctx);

  std::cout.rdbuf(old); // restore
  EXPECT_EQ(buffer.str(), "5\n");
}

// Integration test
TEST(V1_IntegrationTest, FullProgram) {
  std::string src = "x = 1 + 2 * 3;"
                    "y = x + 10;"
                    "z = (x + y) / 2;";

  Lexer lex(src);
  Parser parser(lex);
  auto program = parser.parseProgram();

  std::map<std::string, int> ctx;
  for (auto &stmt : program) {
    stmt->exec(ctx);
  }

  EXPECT_EQ(ctx["x"], 7);  // 1 + 2 * 3 = 7
  EXPECT_EQ(ctx["y"], 17); // 7 + 10 = 17
  EXPECT_EQ(ctx["z"], 12); // (7 + 17) / 2 = 12
}

TEST(V1_IntegrationTest, ProgramWithPrint) {
  std::string src = "x = 5 * 2;"
                    "print(x);";

  Lexer lex(src);
  Parser parser(lex);
  auto program = parser.parseProgram();

  std::map<std::string, int> ctx;

  // Redirect stdout
  std::stringstream buffer;
  std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

  for (auto &stmt : program) {
    stmt->exec(ctx);
  }

  std::cout.rdbuf(old);

  EXPECT_EQ(ctx["x"], 10);
  EXPECT_EQ(buffer.str(), "10\n");
}

TEST(V1_IntegrationTest, MultipleVariables) {
  std::string src = "a = 10;"
                    "b = 20;"
                    "c = a + b;"
                    "d = c * 2;";

  Lexer lex(src);
  Parser parser(lex);
  auto program = parser.parseProgram();

  std::map<std::string, int> ctx;
  for (auto &stmt : program) {
    stmt->exec(ctx);
  }

  EXPECT_EQ(ctx["a"], 10);
  EXPECT_EQ(ctx["b"], 20);
  EXPECT_EQ(ctx["c"], 30);
  EXPECT_EQ(ctx["d"], 60);
}
} // namespace Interpreter_V1
