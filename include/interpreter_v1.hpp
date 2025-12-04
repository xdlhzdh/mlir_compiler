#pragma once

#include <cctype>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Interpreter_V1 {
// ============================================================================
// Value
// ============================================================================

struct Value {
  enum { INT, BOOL, STRING, NONE } type;
  int i;
  bool b;
  std::string s;

  Value() : type(NONE) {}
  Value(int v) : type(INT), i(v) {}
  Value(bool v) : type(BOOL), b(v) {}
  Value(std::string v) : type(STRING), s(v) {}
};

// ============================================================================
// AST
// ============================================================================

struct ExprAST {
  virtual ~ExprAST() = default;
  virtual int eval(std::map<std::string, int> &ctx) = 0;
};

struct NumberExprAST : ExprAST {
  int value;
  NumberExprAST(int v) : value(v) {}
  int eval(std::map<std::string, int> &) override { return value; }
};

struct VariableExprAST : ExprAST {
  std::string name;
  VariableExprAST(const std::string &n) : name(n) {}
  int eval(std::map<std::string, int> &ctx) override { return ctx[name]; }
};

struct BinaryExprAST : ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;
  BinaryExprAST(char oper, std::unique_ptr<ExprAST> L,
                std::unique_ptr<ExprAST> R)
      : op(oper), lhs(std::move(L)), rhs(std::move(R)) {}

  int eval(std::map<std::string, int> &ctx) override {
    int l = lhs->eval(ctx);
    int r = rhs->eval(ctx);
    switch (op) {
    case '+':
      return l + r;
    case '-':
      return l - r;
    case '*':
      return l * r;
    case '/':
      return l / r;
    }
    return 0;
  }
};

// statement
struct StmtAST {
  virtual ~StmtAST() = default;
  virtual void exec(std::map<std::string, int> &ctx) = 0;
};

struct AssignStmtAST : StmtAST {
  std::string name;
  std::unique_ptr<ExprAST> expr;
  AssignStmtAST(const std::string &n, std::unique_ptr<ExprAST> e)
      : name(n), expr(std::move(e)) {}

  void exec(std::map<std::string, int> &ctx) override {
    ctx[name] = expr->eval(ctx);
  }
};

struct PrintStmtAST : StmtAST {
  std::unique_ptr<ExprAST> expr;
  PrintStmtAST(std::unique_ptr<ExprAST> e) : expr(std::move(e)) {}

  void exec(std::map<std::string, int> &ctx) override {
    std::cout << expr->eval(ctx) << std::endl;
  }
};

// ============================================================================
// Lexer
// ============================================================================

enum TokenType {
  tok_eof = -1,

  tok_number = -2,
  tok_identifier = -3,

  tok_print = -4,

  tok_equal = '=',
  tok_plus = '+',
  tok_minus = '-',
  tok_mul = '*',
  tok_div = '/',
  tok_lparen = '(',
  tok_rparen = ')',
  tok_semicolon = ';',
};

struct Token {
  TokenType type;
  std::string text;
  int number;

  Token(TokenType t, std::string txt = "", int num = 0)
      : type(t), text(txt), number(num) {}
};

class Lexer {
  std::string input;
  int pos = 0;
  int len;

public:
  Lexer(const std::string &src) : input(src), len(src.size()) {}

  char peek() { return pos < len ? input[pos] : '\0'; }
  char get() { return pos < len ? input[pos++] : '\0'; }

  Token nextToken() {
    while (isspace(peek()))
      get();

    char c = peek();
    if (c == '\0')
      return Token(tok_eof);

    // number
    if (isdigit(c)) {
      int v = 0;
      while (isdigit(peek())) {
        v = v * 10 + (get() - '0');
      }
      return Token(tok_number, "", v);
    }

    // identifier / keywords
    if (isalpha(c)) {
      std::string id;
      while (isalnum(peek()))
        id += get();

      if (id == "print")
        return Token(tok_print);

      return Token(tok_identifier, id);
    }

    // single char tokens
    get();
    return Token((TokenType)c);
  }
};

// ============================================================================
// Parser
// ============================================================================

class Parser {
  Lexer &lex;
  Token curTok;

  Token getNext() {
    curTok = lex.nextToken();
    return curTok;
  }

public:
  explicit Parser(Lexer &L) : lex(L), curTok(tok_eof) { getNext(); }

  // factor
  std::unique_ptr<ExprAST> parseFactor() {
    if (curTok.type == tok_number) {
      int v = curTok.number;
      getNext();
      return std::make_unique<NumberExprAST>(v);
    }

    if (curTok.type == tok_identifier) {
      std::string name = curTok.text;
      getNext();
      return std::make_unique<VariableExprAST>(name);
    }

    if (curTok.type == tok_lparen) {
      getNext(); // (
      auto expr = parseExpr();
      if (curTok.type != tok_rparen)
        throw std::runtime_error("missing )");
      getNext();
      return expr;
    }

    throw std::runtime_error("invalid factor");
  }

  // term
  std::unique_ptr<ExprAST> parseTerm() {
    auto node = parseFactor();
    while (curTok.type == tok_mul || curTok.type == tok_div) {
      char op = (char)curTok.type;
      getNext();
      node =
          std::make_unique<BinaryExprAST>(op, std::move(node), parseFactor());
    }
    return node;
  }

  // expr
  std::unique_ptr<ExprAST> parseExpr() {
    auto node = parseTerm();
    while (curTok.type == tok_plus || curTok.type == tok_minus) {
      char op = (char)curTok.type;
      getNext();
      node = std::make_unique<BinaryExprAST>(op, std::move(node), parseTerm());
    }
    return node;
  }

  // stmt
  std::unique_ptr<StmtAST> parseStmt() {
    if (curTok.type == tok_identifier) {
      std::string name = curTok.text;
      getNext();

      if (curTok.type != tok_equal)
        throw std::runtime_error("expected =");
      getNext();

      auto expr = parseExpr();

      if (curTok.type != tok_semicolon)
        throw std::runtime_error("expected ;");
      getNext();

      return std::make_unique<AssignStmtAST>(name, std::move(expr));
    }

    if (curTok.type == tok_print) {
      getNext();
      if (curTok.type != tok_lparen)
        throw std::runtime_error("expected (");
      getNext();

      auto expr = parseExpr();

      if (curTok.type != tok_rparen)
        throw std::runtime_error("expected )");
      getNext();

      if (curTok.type != tok_semicolon)
        throw std::runtime_error("expected ;");
      getNext();

      return std::make_unique<PrintStmtAST>(std::move(expr));
    }

    throw std::runtime_error("invalid statement");
  }

  std::vector<std::unique_ptr<StmtAST>> parseProgram() {
    std::vector<std::unique_ptr<StmtAST>> stmts;
    while (curTok.type != tok_eof) {
      stmts.push_back(parseStmt());
    }
    return stmts;
  }
};

} // namespace Interpreter_V1