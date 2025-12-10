#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace Interpreter_V3 {

// ===== Type System =====
enum class ValueType { NONE, INT, DOUBLE, STRING, BOOL, FUNCTION };

// ===== Forward declarations =====
struct Env;
class Stmt;

// ===== Function/Closure type =====
struct Function {
  std::vector<std::string> params;
  std::shared_ptr<Stmt> body;
  std::shared_ptr<Env> capturedEnv; // For closures
  Function(std::vector<std::string> p, std::shared_ptr<Stmt> b,
           std::shared_ptr<Env> env = nullptr)
      : params(std::move(p)), body(b), capturedEnv(env) {}
};

struct Value {
  ValueType type;
  std::variant<int, double, std::string, bool, std::shared_ptr<Function>> data;

  Value() : type(ValueType::NONE), data(0) {}
  Value(int v) : type(ValueType::INT), data(v) {}
  Value(double v) : type(ValueType::DOUBLE), data(v) {}
  Value(const char *v) : type(ValueType::STRING), data(std::string(v)) {}
  Value(std::string v) : type(ValueType::STRING), data(std::move(v)) {}
  Value(bool v) : type(ValueType::BOOL), data(v) {}
  Value(std::shared_ptr<Function> v) : type(ValueType::FUNCTION), data(v) {}

  bool isNumber() const {
    return type == ValueType::INT || type == ValueType::DOUBLE;
  }

  double toNumber() const {
    if (type == ValueType::INT)
      return static_cast<double>(std::get<int>(data));
    if (type == ValueType::DOUBLE)
      return std::get<double>(data);
    if (type == ValueType::BOOL)
      return std::get<bool>(data) ? 1.0 : 0.0;
    return 0.0;
  }

  int toInt() const {
    if (type == ValueType::INT)
      return std::get<int>(data);
    if (type == ValueType::DOUBLE)
      return static_cast<int>(std::get<double>(data));
    if (type == ValueType::BOOL)
      return std::get<bool>(data) ? 1 : 0;
    return 0;
  }

  bool toBool() const {
    if (type == ValueType::BOOL)
      return std::get<bool>(data);
    if (type == ValueType::INT)
      return std::get<int>(data) != 0;
    if (type == ValueType::DOUBLE)
      return std::get<double>(data) != 0.0;
    if (type == ValueType::STRING)
      return !std::get<std::string>(data).empty();
    return false;
  }

  std::string toString() const {
    if (type == ValueType::STRING)
      return std::get<std::string>(data);
    if (type == ValueType::INT)
      return std::to_string(std::get<int>(data));
    if (type == ValueType::DOUBLE)
      return std::to_string(std::get<double>(data));
    if (type == ValueType::BOOL)
      return std::get<bool>(data) ? "true" : "false";
    return "none";
  }
};

// ===== Environment with scope chain =====
struct Env {
  std::unordered_map<std::string, Value> vars;
  std::shared_ptr<Env> parent;

  Env(std::shared_ptr<Env> p = nullptr) : parent(p) {}

  void define(const std::string &name, Value val) { vars[name] = val; }

  Value &get(const std::string &name) {
    if (vars.count(name))
      return vars[name];
    if (parent)
      return parent->get(name);
    throw std::runtime_error("Undefined variable: " + name);
  }

  bool has(const std::string &name) const {
    if (vars.count(name))
      return true;
    if (parent)
      return parent->has(name);
    return false;
  }

  void set(const std::string &name, Value val) {
    if (vars.count(name)) {
      vars[name] = val;
      return;
    }
    if (parent) {
      parent->set(name, val);
      return;
    }
    throw std::runtime_error("Undefined variable: " + name);
  }
};

// ===== Lexer =====
enum TokenType {
  TOK_EOF,          // End of file
  TOK_ID,           // Identifier (e.g., variable or function name)
  TOK_INT,          // Integer literal
  TOK_DOUBLE,       // Double (floating-point) literal
  TOK_STR,          // String literal
  TOK_TRUE,         // Boolean literal 'true'
  TOK_FALSE,        // Boolean literal 'false'
  TOK_LET,          // 'let' keyword for variable declaration
  TOK_CONST,        // 'const' keyword for constant declaration
  TOK_FN,           // 'fn' keyword for function definition
  TOK_RETURN,       // 'return' keyword for returning a value
  TOK_IF,           // 'if' keyword for conditional statements
  TOK_ELSE,         // 'else' keyword for alternative branch in conditionals
  TOK_WHILE,        // 'while' keyword for loops
  TOK_FOR,          // 'for' keyword for loops
  TOK_PRINT,        // 'print' keyword for output
  TOK_TYPEOF,       // 'typeof' keyword for type checking
  TOK_PLUS,         // Addition operator (+)
  TOK_MINUS,        // Subtraction operator (-)
  TOK_MUL,          // Multiplication operator (*)
  TOK_DIV,          // Division operator (/)
  TOK_MOD,          // Modulo operator (%)
  TOK_EQ,           // Equality operator (==)
  TOK_NEQ,          // Inequality operator (!=)
  TOK_LT,           // Less than operator (<)
  TOK_GT,           // Greater than operator (>)
  TOK_LE,           // Less than or equal to operator (<=)
  TOK_GE,           // Greater than or equal to operator (>=)
  TOK_AND,          // Logical AND operator (&&)
  TOK_OR,           // Logical OR operator (||)
  TOK_NOT,          // Logical NOT operator (!)
  TOK_ASSIGN,       // Assignment operator (=)
  TOK_PLUS_ASSIGN,  // Compound addition assignment operator (+=)
  TOK_MINUS_ASSIGN, // Compound subtraction assignment operator (-=)
  TOK_MUL_ASSIGN,   // Compound multiplication assignment operator (*=)
  TOK_DIV_ASSIGN,   // Compound division assignment operator (/=)
  TOK_LP,           // Left parenthesis '('
  TOK_RP,           // Right parenthesis ')'
  TOK_LB,           // Left brace '{'
  TOK_RB,           // Right brace '}'
  TOK_SEMI,         // Semicolon ';'
  TOK_COMMA,        // Comma ','
  TOK_QUESTION,     // Question mark '?'
  TOK_COLON,        // Colon ':'
  TOK_ARROW         // Arrow '=>', used in lambda expressions
};

struct Token {
  TokenType type;
  std::string text;
  Token(TokenType t = TOK_EOF, std::string s = "")
      : type(t), text(std::move(s)) {}
};

class Lexer {
  std::string src;
  size_t pos;

  bool isIdStart(char c) { return std::isalpha(c) || c == '_'; }
  bool isIdChar(char c) { return std::isalnum(c) || c == '_'; }

public:
  Lexer(const std::string &source) : src(source), pos(0) {}

  Token nextToken() {
    while (pos < src.size() && std::isspace(src[pos]))
      pos++;

    if (pos >= src.size())
      return Token(TOK_EOF);

    char c = src[pos];

    // Numbers (int and double)
    if (std::isdigit(c)) {
      size_t start = pos;
      bool isDouble = false;
      while (pos < src.size() && (std::isdigit(src[pos]) || src[pos] == '.')) {
        if (src[pos] == '.')
          isDouble = true;
        pos++;
      }
      std::string num = src.substr(start, pos - start);
      return Token(isDouble ? TOK_DOUBLE : TOK_INT, num);
    }

    // Identifiers and keywords
    if (isIdStart(c)) {
      size_t start = pos;
      while (pos < src.size() && isIdChar(src[pos]))
        pos++;
      std::string id = src.substr(start, pos - start);

      if (id == "let")
        return Token(TOK_LET, id);
      if (id == "const")
        return Token(TOK_CONST, id);
      if (id == "fn")
        return Token(TOK_FN, id);
      if (id == "return")
        return Token(TOK_RETURN, id);
      if (id == "if")
        return Token(TOK_IF, id);
      if (id == "else")
        return Token(TOK_ELSE, id);
      if (id == "while")
        return Token(TOK_WHILE, id);
      if (id == "for")
        return Token(TOK_FOR, id);
      if (id == "print")
        return Token(TOK_PRINT, id);
      if (id == "typeof")
        return Token(TOK_TYPEOF, id);
      if (id == "true")
        return Token(TOK_TRUE, id);
      if (id == "false")
        return Token(TOK_FALSE, id);

      return Token(TOK_ID, id);
    }

    // String literals
    if (c == '"') {
      pos++;
      size_t start = pos;
      while (pos < src.size() && src[pos] != '"')
        pos++;
      std::string str = src.substr(start, pos - start);
      if (pos < src.size())
        pos++; // consume closing "
      return Token(TOK_STR, str);
    }

    // Two-character operators
    if (pos + 1 < src.size()) {
      std::string two = src.substr(pos, 2);
      if (two == "==") {
        pos += 2;
        return Token(TOK_EQ, two);
      }
      if (two == "!=") {
        pos += 2;
        return Token(TOK_NEQ, two);
      }
      if (two == "<=") {
        pos += 2;
        return Token(TOK_LE, two);
      }
      if (two == ">=") {
        pos += 2;
        return Token(TOK_GE, two);
      }
      if (two == "&&") {
        pos += 2;
        return Token(TOK_AND, two);
      }
      if (two == "||") {
        pos += 2;
        return Token(TOK_OR, two);
      }
      if (two == "+=") {
        pos += 2;
        return Token(TOK_PLUS_ASSIGN, two);
      }
      if (two == "-=") {
        pos += 2;
        return Token(TOK_MINUS_ASSIGN, two);
      }
      if (two == "*=") {
        pos += 2;
        return Token(TOK_MUL_ASSIGN, two);
      }
      if (two == "/=") {
        pos += 2;
        return Token(TOK_DIV_ASSIGN, two);
      }
      if (two == "=>") {
        pos += 2;
        return Token(TOK_ARROW, two);
      }
    }

    // Single character tokens
    pos++;
    switch (c) {
    case '+':
      return Token(TOK_PLUS, "+");
    case '-':
      return Token(TOK_MINUS, "-");
    case '*':
      return Token(TOK_MUL, "*");
    case '/':
      return Token(TOK_DIV, "/");
    case '%':
      return Token(TOK_MOD, "%");
    case '<':
      return Token(TOK_LT, "<");
    case '>':
      return Token(TOK_GT, ">");
    case '!':
      return Token(TOK_NOT, "!");
    case '=':
      return Token(TOK_ASSIGN, "=");
    case '(':
      return Token(TOK_LP, "(");
    case ')':
      return Token(TOK_RP, ")");
    case '{':
      return Token(TOK_LB, "{");
    case '}':
      return Token(TOK_RB, "}");
    case ';':
      return Token(TOK_SEMI, ";");
    case ',':
      return Token(TOK_COMMA, ",");
    case '?':
      return Token(TOK_QUESTION, "?");
    case ':':
      return Token(TOK_COLON, ":");
    default:
      throw std::runtime_error("Unknown character: " + std::string(1, c));
    }
  }
};

// ===== Expression AST =====
class Expr {
public:
  virtual ~Expr() = default;
  virtual Value eval(std::shared_ptr<Env> env) = 0;
};

class LiteralExpr : public Expr {
  Value value;

public:
  LiteralExpr(Value v) : value(v) {}
  Value eval(std::shared_ptr<Env>) override { return value; }
};

class VarExpr : public Expr {
  std::string name;

public:
  VarExpr(std::string n) : name(std::move(n)) {}
  Value eval(std::shared_ptr<Env> env) override { return env->get(name); }
};

class BinaryExpr : public Expr {
  std::string op;
  std::unique_ptr<Expr> left, right;

public:
  BinaryExpr(std::string o, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r)
      : op(std::move(o)), left(std::move(l)), right(std::move(r)) {}

  Value eval(std::shared_ptr<Env> env) override {
    Value lval = left->eval(env);
    Value rval = right->eval(env);

    // Arithmetic
    if (op == "+") {
      if (lval.type == ValueType::STRING || rval.type == ValueType::STRING)
        return Value(lval.toString() + rval.toString());
      return Value(lval.toNumber() + rval.toNumber());
    }
    if (op == "-")
      return Value(lval.toNumber() - rval.toNumber());
    if (op == "*")
      return Value(lval.toNumber() * rval.toNumber());
    if (op == "/")
      return Value(lval.toNumber() / rval.toNumber());
    if (op == "%")
      return Value(lval.toInt() % rval.toInt());

    // Comparison
    if (op == "==") {
      if (lval.type == ValueType::STRING && rval.type == ValueType::STRING)
        return Value(lval.toString() == rval.toString());
      return Value(lval.toNumber() == rval.toNumber());
    }
    if (op == "!=") {
      if (lval.type == ValueType::STRING && rval.type == ValueType::STRING)
        return Value(lval.toString() != rval.toString());
      return Value(lval.toNumber() != rval.toNumber());
    }
    if (op == "<")
      return Value(lval.toNumber() < rval.toNumber());
    if (op == ">")
      return Value(lval.toNumber() > rval.toNumber());
    if (op == "<=")
      return Value(lval.toNumber() <= rval.toNumber());
    if (op == ">=")
      return Value(lval.toNumber() >= rval.toNumber());

    // Logical
    if (op == "&&")
      return Value(lval.toBool() && rval.toBool());
    if (op == "||")
      return Value(lval.toBool() || rval.toBool());

    throw std::runtime_error("Unknown binary operator: " + op);
  }
};

class UnaryExpr : public Expr {
  std::string op;
  std::unique_ptr<Expr> operand;

public:
  UnaryExpr(std::string o, std::unique_ptr<Expr> e)
      : op(std::move(o)), operand(std::move(e)) {}

  Value eval(std::shared_ptr<Env> env) override {
    Value val = operand->eval(env);
    if (op == "-")
      return Value(-val.toNumber());
    if (op == "!")
      return Value(!val.toBool());
    throw std::runtime_error("Unknown unary operator: " + op);
  }
};

// Ternary conditional operator: condition ? true_expr : false_expr
class TernaryExpr : public Expr {
  std::unique_ptr<Expr> condition, trueExpr, falseExpr;

public:
  TernaryExpr(std::unique_ptr<Expr> c, std::unique_ptr<Expr> t,
              std::unique_ptr<Expr> f)
      : condition(std::move(c)), trueExpr(std::move(t)),
        falseExpr(std::move(f)) {}

  Value eval(std::shared_ptr<Env> env) override {
    return condition->eval(env).toBool() ? trueExpr->eval(env)
                                         : falseExpr->eval(env);
  }
};

class CallExpr : public Expr {
  std::unique_ptr<Expr> callee;
  std::vector<std::unique_ptr<Expr>> args;

public:
  CallExpr(std::unique_ptr<Expr> c, std::vector<std::unique_ptr<Expr>> a)
      : callee(std::move(c)), args(std::move(a)) {}

  Value eval(std::shared_ptr<Env> env) override;
};

class FunctionExpr : public Expr {
  std::vector<std::string> params;
  std::shared_ptr<Stmt> body; // Changed to shared_ptr

public:
  FunctionExpr(std::vector<std::string> p, std::unique_ptr<Stmt> b)
      : params(std::move(p)), body(std::move(b)) {}

  Value eval(std::shared_ptr<Env> env) override {
    // Capture current environment for closure and share the body
    auto func = std::make_shared<Function>(params, body, env);
    return Value(func);
  }
};

class TypeofExpr : public Expr {
  std::unique_ptr<Expr> operand;

public:
  TypeofExpr(std::unique_ptr<Expr> e) : operand(std::move(e)) {}

  Value eval(std::shared_ptr<Env> env) override {
    Value val = operand->eval(env);
    switch (val.type) {
    case ValueType::INT:
      return Value("int");
    case ValueType::DOUBLE:
      return Value("double");
    case ValueType::STRING:
      return Value("string");
    case ValueType::BOOL:
      return Value("bool");
    case ValueType::FUNCTION:
      return Value("function");
    default:
      return Value("none");
    }
  }
};

// ===== Statement AST =====
struct ReturnException {
  Value value;
  ReturnException(Value v) : value(v) {}
};

class Stmt {
public:
  virtual ~Stmt() = default;
  virtual void exec(std::shared_ptr<Env> env) = 0;
};

class ExprStmt : public Stmt {
  std::unique_ptr<Expr> expr;

public:
  ExprStmt(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
  void exec(std::shared_ptr<Env> env) override { expr->eval(env); }
};

class BlockStmt : public Stmt {
public:
  std::vector<std::unique_ptr<Stmt>> statements;

  void exec(std::shared_ptr<Env> env) override {
    auto blockEnv = std::make_shared<Env>(env);
    for (auto &stmt : statements)
      stmt->exec(blockEnv);
  }
};

class VarDeclStmt : public Stmt {
  std::string name;
  std::unique_ptr<Expr> init;
  bool isConst;

public:
  VarDeclStmt(std::string n, std::unique_ptr<Expr> i, bool c = false)
      : name(std::move(n)), init(std::move(i)), isConst(c) {}

  void exec(std::shared_ptr<Env> env) override {
    Value val = init ? init->eval(env) : Value();
    env->define(name, val);
  }
};

class AssignStmt : public Stmt {
  std::string name;
  std::unique_ptr<Expr> value;
  std::string op; // =, +=, -=, *=, /=

public:
  AssignStmt(std::string n, std::unique_ptr<Expr> v, std::string o = "=")
      : name(std::move(n)), value(std::move(v)), op(std::move(o)) {}

  void exec(std::shared_ptr<Env> env) override {
    Value newVal = value->eval(env);

    if (op == "=") {
      env->set(name, newVal);
    } else {
      Value oldVal = env->get(name);
      if (op == "+=") {
        if (oldVal.type == ValueType::STRING ||
            newVal.type == ValueType::STRING)
          env->set(name, Value(oldVal.toString() + newVal.toString()));
        else
          env->set(name, Value(oldVal.toNumber() + newVal.toNumber()));
      } else if (op == "-=")
        env->set(name, Value(oldVal.toNumber() - newVal.toNumber()));
      else if (op == "*=")
        env->set(name, Value(oldVal.toNumber() * newVal.toNumber()));
      else if (op == "/=")
        env->set(name, Value(oldVal.toNumber() / newVal.toNumber()));
    }
  }
};

class PrintStmt : public Stmt {
  std::vector<std::unique_ptr<Expr>> exprs;

public:
  PrintStmt(std::vector<std::unique_ptr<Expr>> e) : exprs(std::move(e)) {}

  void exec(std::shared_ptr<Env> env) override {
    for (size_t i = 0; i < exprs.size(); i++) {
      Value val = exprs[i]->eval(env);
      std::cout << val.toString();
      if (i < exprs.size() - 1)
        std::cout << " ";
    }
    std::cout << std::endl;
  }
};

class IfStmt : public Stmt {
  std::unique_ptr<Expr> condition;
  std::unique_ptr<Stmt> thenStmt, elseStmt;

public:
  IfStmt(std::unique_ptr<Expr> c, std::unique_ptr<Stmt> t,
         std::unique_ptr<Stmt> e = nullptr)
      : condition(std::move(c)), thenStmt(std::move(t)),
        elseStmt(std::move(e)) {}

  void exec(std::shared_ptr<Env> env) override {
    if (condition->eval(env).toBool())
      thenStmt->exec(env);
    else if (elseStmt)
      elseStmt->exec(env);
  }
};

class WhileStmt : public Stmt {
  std::unique_ptr<Expr> condition;
  std::unique_ptr<Stmt> body;

public:
  WhileStmt(std::unique_ptr<Expr> c, std::unique_ptr<Stmt> b)
      : condition(std::move(c)), body(std::move(b)) {}

  void exec(std::shared_ptr<Env> env) override {
    while (condition->eval(env).toBool())
      body->exec(env);
  }
};

class ReturnStmt : public Stmt {
  std::unique_ptr<Expr> value;

public:
  ReturnStmt(std::unique_ptr<Expr> v = nullptr) : value(std::move(v)) {}

  void exec(std::shared_ptr<Env> env) override {
    Value val = value ? value->eval(env) : Value();
    throw ReturnException(val);
  }
};

class FunctionStmt : public Stmt {
  std::string name;
  std::vector<std::string> params;
  std::shared_ptr<Stmt> body; // Changed to shared_ptr so it can be shared

public:
  FunctionStmt(std::string n, std::vector<std::string> p,
               std::unique_ptr<Stmt> b)
      : name(std::move(n)), params(std::move(p)), body(std::move(b)) {}

  void exec(std::shared_ptr<Env> env) override {
    // Share the body with the Function object
    auto func = std::make_shared<Function>(params, body, env);
    env->define(name, Value(func));
  }
};

// CallExpr implementation
Value CallExpr::eval(std::shared_ptr<Env> env) {
  Value funcVal = callee->eval(env);
  if (funcVal.type != ValueType::FUNCTION)
    throw std::runtime_error("Not a function");

  auto func = std::get<std::shared_ptr<Function>>(funcVal.data);
  if (args.size() != func->params.size())
    throw std::runtime_error("Argument count mismatch");

  // Create new environment with captured environment as parent (closure)
  auto callEnv = std::make_shared<Env>(func->capturedEnv);
  for (size_t i = 0; i < args.size(); i++) {
    callEnv->define(func->params[i], args[i]->eval(env));
  }

  try {
    func->body->exec(callEnv);
    return Value(); // No explicit return
  } catch (ReturnException &e) {
    return e.value;
  }
}

// ===== Parser =====
class Parser {
  Lexer lexer;
  Token curTok;

  void advance() { curTok = lexer.nextToken(); }

  bool match(TokenType type) {
    if (curTok.type == type) {
      advance();
      return true;
    }
    return false;
  }

  void expect(TokenType type, const std::string &msg = "Unexpected token") {
    if (curTok.type != type)
      throw std::runtime_error(msg + ", got: " + curTok.text);
    advance();
  }

public:
  Parser(const std::string &source) : lexer(source) { advance(); }

  // Expression parsing with operator precedence
  std::unique_ptr<Expr> parseExpression() { return parseTernary(); }

  std::unique_ptr<Expr> parseTernary() {
    auto expr = parseLogicalOr();
    if (match(TOK_QUESTION)) {
      auto trueExpr = parseExpression();
      expect(TOK_COLON, "Expected ':' in ternary operator");
      auto falseExpr = parseTernary();
      return std::make_unique<TernaryExpr>(std::move(expr), std::move(trueExpr),
                                           std::move(falseExpr));
    }
    return expr;
  }

  std::unique_ptr<Expr> parseLogicalOr() {
    auto left = parseLogicalAnd();
    while (match(TOK_OR)) {
      auto right = parseLogicalAnd();
      left =
          std::make_unique<BinaryExpr>("||", std::move(left), std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseLogicalAnd() {
    auto left = parseEquality();
    while (match(TOK_AND)) {
      auto right = parseEquality();
      left =
          std::make_unique<BinaryExpr>("&&", std::move(left), std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseEquality() {
    auto left = parseRelational();
    while (curTok.type == TOK_EQ || curTok.type == TOK_NEQ) {
      std::string op = curTok.text;
      advance();
      auto right = parseRelational();
      left =
          std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseRelational() {
    auto left = parseAdditive();
    while (curTok.type == TOK_LT || curTok.type == TOK_GT ||
           curTok.type == TOK_LE || curTok.type == TOK_GE) {
      std::string op = curTok.text;
      advance();
      auto right = parseAdditive();
      left =
          std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseAdditive() {
    auto left = parseMultiplicative();
    while (curTok.type == TOK_PLUS || curTok.type == TOK_MINUS) {
      std::string op = curTok.text;
      advance();
      auto right = parseMultiplicative();
      left =
          std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseMultiplicative() {
    auto left = parseUnary();
    while (curTok.type == TOK_MUL || curTok.type == TOK_DIV ||
           curTok.type == TOK_MOD) {
      std::string op = curTok.text;
      advance();
      auto right = parseUnary();
      left =
          std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }
    return left;
  }

  std::unique_ptr<Expr> parseUnary() {
    if (curTok.type == TOK_MINUS || curTok.type == TOK_NOT) {
      std::string op = curTok.text;
      advance();
      return std::make_unique<UnaryExpr>(op, parseUnary());
    }
    if (curTok.type == TOK_TYPEOF) {
      advance();
      return std::make_unique<TypeofExpr>(parseUnary());
    }
    return parsePostfix();
  }

  std::unique_ptr<Expr> parsePostfix() {
    auto expr = parsePrimary();
    while (curTok.type == TOK_LP) {
      advance();
      std::vector<std::unique_ptr<Expr>> args;
      if (curTok.type != TOK_RP) {
        args.push_back(parseExpression());
        while (match(TOK_COMMA))
          args.push_back(parseExpression());
      }
      expect(TOK_RP, "Expected ')' after arguments");
      expr = std::make_unique<CallExpr>(std::move(expr), std::move(args));
    }
    return expr;
  }

  std::unique_ptr<Expr> parsePrimary() {
    // Literals
    if (curTok.type == TOK_INT) {
      int val = std::stoi(curTok.text);
      advance();
      return std::make_unique<LiteralExpr>(Value(val));
    }
    if (curTok.type == TOK_DOUBLE) {
      double val = std::stod(curTok.text);
      advance();
      return std::make_unique<LiteralExpr>(Value(val));
    }
    if (curTok.type == TOK_STR) {
      std::string val = curTok.text;
      advance();
      return std::make_unique<LiteralExpr>(Value(val));
    }
    if (curTok.type == TOK_TRUE) {
      advance();
      return std::make_unique<LiteralExpr>(Value(true));
    }
    if (curTok.type == TOK_FALSE) {
      advance();
      return std::make_unique<LiteralExpr>(Value(false));
    }

    // Identifiers
    if (curTok.type == TOK_ID) {
      std::string name = curTok.text;
      advance();
      return std::make_unique<VarExpr>(name);
    }

    // Parenthesized expression
    if (match(TOK_LP)) {
      auto expr = parseExpression();
      expect(TOK_RP, "Expected ')' after expression");
      return expr;
    }

    // Lambda/Arrow function: (params) => body or param => body
    if (curTok.type == TOK_LP || curTok.type == TOK_ID) {
      size_t savePos = lexer.nextToken().type; // lookahead
      // Simple check for arrow function
      // For now, handle in function declaration
    }

    throw std::runtime_error("Unexpected token in expression: " + curTok.text);
  }

  // Statement parsing
  std::unique_ptr<Stmt> parseStatement() {
    if (curTok.type == TOK_LET || curTok.type == TOK_CONST)
      return parseVarDecl();
    if (curTok.type == TOK_FN)
      return parseFunctionDecl();
    if (curTok.type == TOK_IF)
      return parseIfStmt();
    if (curTok.type == TOK_WHILE)
      return parseWhileStmt();
    if (curTok.type == TOK_PRINT)
      return parsePrintStmt();
    if (curTok.type == TOK_RETURN)
      return parseReturnStmt();
    if (curTok.type == TOK_LB)
      return parseBlock();

    // Assignment or expression statement
    if (curTok.type == TOK_ID) {
      std::string name = curTok.text;
      advance();
      if (curTok.type == TOK_ASSIGN || curTok.type == TOK_PLUS_ASSIGN ||
          curTok.type == TOK_MINUS_ASSIGN || curTok.type == TOK_MUL_ASSIGN ||
          curTok.type == TOK_DIV_ASSIGN) {
        std::string op = curTok.text;
        advance();
        auto value = parseExpression();
        expect(TOK_SEMI, "Expected ';' after assignment");
        return std::make_unique<AssignStmt>(name, std::move(value), op);
      }
      // Not assignment, it's an expression (maybe function call)
      auto varExpr = std::make_unique<VarExpr>(name);
      if (curTok.type == TOK_LP) {
        advance();
        std::vector<std::unique_ptr<Expr>> args;
        if (curTok.type != TOK_RP) {
          args.push_back(parseExpression());
          while (match(TOK_COMMA))
            args.push_back(parseExpression());
        }
        expect(TOK_RP, "Expected ')' after arguments");
        auto callExpr =
            std::make_unique<CallExpr>(std::move(varExpr), std::move(args));
        expect(TOK_SEMI, "Expected ';' after expression");
        return std::make_unique<ExprStmt>(std::move(callExpr));
      }
      expect(TOK_SEMI, "Expected ';' after expression");
      return std::make_unique<ExprStmt>(std::move(varExpr));
    }

    auto expr = parseExpression();
    expect(TOK_SEMI, "Expected ';' after expression");
    return std::make_unique<ExprStmt>(std::move(expr));
  }

  std::unique_ptr<Stmt> parseVarDecl() {
    bool isConst = curTok.type == TOK_CONST;
    advance();
    if (curTok.type != TOK_ID)
      throw std::runtime_error("Expected variable name after let/const");
    std::string name = curTok.text;
    advance();
    if (curTok.type != TOK_ASSIGN)
      throw std::runtime_error("Expected '=' in variable declaration");
    advance();
    auto init = parseExpression();
    expect(TOK_SEMI, "Expected ';' after variable declaration");
    return std::make_unique<VarDeclStmt>(name, std::move(init), isConst);
  }

  std::unique_ptr<Stmt> parseFunctionDecl() {
    advance(); // consume 'fn'
    if (curTok.type != TOK_ID)
      throw std::runtime_error("Expected function name");
    std::string name = curTok.text;
    advance();
    if (curTok.type != TOK_LP)
      throw std::runtime_error("Expected '(' after function name");
    advance();

    std::vector<std::string> params;
    if (curTok.type != TOK_RP) {
      if (curTok.type != TOK_ID)
        throw std::runtime_error("Expected parameter name");
      params.push_back(curTok.text);
      advance();
      while (match(TOK_COMMA)) {
        if (curTok.type != TOK_ID)
          throw std::runtime_error("Expected parameter name");
        params.push_back(curTok.text);
        advance();
      }
    }
    if (curTok.type != TOK_RP)
      throw std::runtime_error("Expected ')' after parameters");
    advance();

    auto body = parseBlock();
    return std::make_unique<FunctionStmt>(name, params, std::move(body));
  }

  std::unique_ptr<Stmt> parseIfStmt() {
    advance(); // consume 'if'
    expect(TOK_LP, "Expected '(' after 'if'");
    auto condition = parseExpression();
    expect(TOK_RP, "Expected ')' after condition");
    auto thenStmt = parseStatement();
    std::unique_ptr<Stmt> elseStmt = nullptr;
    if (match(TOK_ELSE))
      elseStmt = parseStatement();
    return std::make_unique<IfStmt>(std::move(condition), std::move(thenStmt),
                                    std::move(elseStmt));
  }

  std::unique_ptr<Stmt> parseWhileStmt() {
    advance(); // consume 'while'
    expect(TOK_LP, "Expected '(' after 'while'");
    auto condition = parseExpression();
    expect(TOK_RP, "Expected ')' after condition");
    auto body = parseStatement();
    return std::make_unique<WhileStmt>(std::move(condition), std::move(body));
  }

  std::unique_ptr<Stmt> parsePrintStmt() {
    advance(); // consume 'print'
    bool hasParen = match(TOK_LP);
    std::vector<std::unique_ptr<Expr>> exprs;
    exprs.push_back(parseExpression());
    while (match(TOK_COMMA))
      exprs.push_back(parseExpression());
    if (hasParen)
      expect(TOK_RP, "Expected ')' after print arguments");
    expect(TOK_SEMI, "Expected ';' after print statement");
    return std::make_unique<PrintStmt>(std::move(exprs));
  }

  std::unique_ptr<Stmt> parseReturnStmt() {
    advance(); // consume 'return'
    std::unique_ptr<Expr> value = nullptr;
    if (curTok.type != TOK_SEMI)
      value = parseExpression();
    expect(TOK_SEMI, "Expected ';' after return statement");
    return std::make_unique<ReturnStmt>(std::move(value));
  }

  std::unique_ptr<Stmt> parseBlock() {
    expect(TOK_LB, "Expected '{'");
    auto block = std::make_unique<BlockStmt>();
    while (curTok.type != TOK_RB && curTok.type != TOK_EOF)
      block->statements.push_back(parseStatement());
    expect(TOK_RB, "Expected '}'");
    return block;
  }

  std::vector<std::unique_ptr<Stmt>> parseProgram() {
    std::vector<std::unique_ptr<Stmt>> statements;
    while (curTok.type != TOK_EOF)
      statements.push_back(parseStatement());
    return statements;
  }
};

} // namespace Interpreter_V3
