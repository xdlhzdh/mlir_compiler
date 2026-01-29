#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef USE_LLVM_CODEGEN
#include <llvm/IR/Constant.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#endif

namespace Interpreter_V4 {

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

#ifdef USE_LLVM_CODEGEN
// ===== LLVM CodeGen Context =====
struct CodeGenContext {
  llvm::IRBuilder<> &builder;
  llvm::Module *module;
  llvm::LLVMContext &context;
  std::vector<std::unordered_map<std::string, llvm::AllocaInst *>> namedValues;
  llvm::Function *currentFunction = nullptr;
  std::unordered_map<std::string, llvm::Function *> functions;
  struct LoopLabels {
    llvm::BasicBlock *cond = nullptr;
    llvm::BasicBlock *exit = nullptr;
  };
  std::vector<LoopLabels> loopStack;

  llvm::Type *doubleTy() { return llvm::Type::getDoubleTy(context); }
  llvm::Type *i1Ty() { return llvm::Type::getInt1Ty(context); }
  llvm::Type *i32Ty() { return llvm::Type::getInt32Ty(context); }
  llvm::Type *i8PtrTy() {
    return llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0);
  }

  void pushScope() { namedValues.emplace_back(); }
  void popScope() { namedValues.pop_back(); }
  llvm::AllocaInst *findAlloca(const std::string &name) {
    for (auto it = namedValues.rbegin(); it != namedValues.rend(); ++it) {
      auto i = it->find(name);
      if (i != it->end())
        return i->second;
    }
    return nullptr;
  }
  void defineAlloca(const std::string &name, llvm::AllocaInst *alloca) {
    namedValues.back()[name] = alloca;
  }
};
#endif

// ===== Lexer =====
enum TokenType {
  TOK_EOF,         // End of file
  TOK_ID,          // Identifier (e.g., variable or function name)
  TOK_INT,         // Integer literal
  TOK_DOUBLE,      // Double (floating-point) literal
  TOK_STR,         // String literal
  TOK_TRUE,        // Boolean literal 'true'
  TOK_FALSE,       // Boolean literal 'false'
  TOK_LET,         // 'let' keyword for variable declaration
  TOK_CONST,       // 'const' keyword for constant declaration
  TOK_FN,          // 'fn' keyword for function definition
  TOK_RETURN,      // 'return' keyword for returning a value
  TOK_IF,          // 'if' keyword for conditional statements
  TOK_ELSE,        // 'else' keyword for alternative branch in conditionals
  TOK_WHILE,       // 'while' keyword for loops
  TOK_FOR,         // 'for' keyword for loops
  TOK_BREAK,       // 'break' keyword to exit loops
  TOK_CONTINUE,    // 'continue' keyword to skip to the next iteration of a loop
  TOK_PRINT,       // 'print' keyword for output
  TOK_TYPEOF,      // 'typeof' keyword for type checking
  TOK_PLUS,        // Addition operator (+)
  TOK_MINUS,       // Subtraction operator (-)
  TOK_MUL,         // Multiplication operator (*)
  TOK_DIV,         // Division operator (/)
  TOK_MOD,         // Modulo operator (%)
  TOK_EQ,          // Equality operator (==)
  TOK_NEQ,         // Inequality operator (!=)
  TOK_LT,          // Less than operator (<)
  TOK_GT,          // Greater than operator (>)
  TOK_LE,          // Less than or equal to operator (<=)
  TOK_GE,          // Greater than or equal to operator (>=)
  TOK_AND,         // Logical AND operator (&&)
  TOK_OR,          // Logical OR operator (||)
  TOK_NOT,         // Logical NOT operator (!)
  TOK_ASSIGN,      // Assignment operator (=)
  TOK_PLUS_ASSIGN, // Compound addition assignment operator (+=)
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
      if (id == "break")
        return Token(TOK_BREAK, id);
      if (id == "continue")
        return Token(TOK_CONTINUE, id);
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
#ifdef USE_LLVM_CODEGEN
  virtual llvm::Value *codegen(CodeGenContext &ctx) { return nullptr; }
#endif
};

class LiteralExpr : public Expr {
  Value value;

public:
  LiteralExpr(Value v) : value(v) {}
  Value eval(std::shared_ptr<Env>) override { return value; }
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &ctx) override {
    switch (value.type) {
    case ValueType::INT:
      return llvm::ConstantFP::get(ctx.doubleTy(),
                                   static_cast<double>(value.toInt()));
    case ValueType::DOUBLE:
      return llvm::ConstantFP::get(ctx.doubleTy(), value.toNumber());
    case ValueType::BOOL:
      return llvm::ConstantInt::get(ctx.i1Ty(), value.toBool() ? 1 : 0);
    default:
      return llvm::ConstantFP::get(ctx.doubleTy(), 0.0);
    }
  }
#endif
};

class VarExpr : public Expr {
  std::string name;

public:
  VarExpr(std::string n) : name(std::move(n)) {}
  const std::string &getName() const { return name; }
  Value eval(std::shared_ptr<Env> env) override { return env->get(name); }
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &ctx) override {
    llvm::AllocaInst *alloca = ctx.findAlloca(name);
    if (!alloca)
      throw std::runtime_error("Unknown variable: " + name);
    return ctx.builder.CreateLoad(ctx.doubleTy(), alloca, name);
  }
#endif
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
      // Keep as INT if both operands are INT
      if (lval.type == ValueType::INT && rval.type == ValueType::INT)
        return Value(lval.toInt() + rval.toInt());
      return Value(lval.toNumber() + rval.toNumber());
    }
    if (op == "-") {
      // Keep as INT if both operands are INT
      if (lval.type == ValueType::INT && rval.type == ValueType::INT)
        return Value(lval.toInt() - rval.toInt());
      return Value(lval.toNumber() - rval.toNumber());
    }
    if (op == "*") {
      // Keep as INT if both operands are INT
      if (lval.type == ValueType::INT && rval.type == ValueType::INT)
        return Value(lval.toInt() * rval.toInt());
      return Value(lval.toNumber() * rval.toNumber());
    }
    if (op == "/") {
      // Keep as INT if both operands are INT
      if (lval.type == ValueType::INT && rval.type == ValueType::INT)
        return Value(lval.toInt() / rval.toInt());
      return Value(lval.toNumber() / rval.toNumber());
    }
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
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &ctx) override {
    llvm::Value *L = left->codegen(ctx);
    llvm::Value *R = right->codegen(ctx);
    if (!L || !R)
      return nullptr;
    bool lIsI1 = L->getType()->isIntegerTy(1);
    bool rIsI1 = R->getType()->isIntegerTy(1);
    if (op == "+" || op == "-" || op == "*" || op == "/") {
      llvm::Value *lD = lIsI1 ? ctx.builder.CreateUIToFP(L, ctx.doubleTy()) : L;
      llvm::Value *rD = rIsI1 ? ctx.builder.CreateUIToFP(R, ctx.doubleTy()) : R;
      if (op == "+")
        return ctx.builder.CreateFAdd(lD, rD, "add");
      if (op == "-")
        return ctx.builder.CreateFSub(lD, rD, "sub");
      if (op == "*")
        return ctx.builder.CreateFMul(lD, rD, "mul");
      if (op == "/")
        return ctx.builder.CreateFDiv(lD, rD, "div");
    }
    if (op == "%") {
      llvm::Value *lI = lIsI1 ? ctx.builder.CreateUIToFP(L, ctx.doubleTy()) : L;
      llvm::Value *rI = rIsI1 ? ctx.builder.CreateUIToFP(R, ctx.doubleTy()) : R;
      lI = ctx.builder.CreateFPToSI(lI, ctx.i32Ty());
      rI = ctx.builder.CreateFPToSI(rI, ctx.i32Ty());
      llvm::Value *rem = ctx.builder.CreateSRem(lI, rI, "rem");
      return ctx.builder.CreateSIToFP(rem, ctx.doubleTy());
    }
    if (op == "==" || op == "!=" || op == "<" || op == ">" || op == "<=" ||
        op == ">=") {
      llvm::Value *lD = lIsI1 ? ctx.builder.CreateUIToFP(L, ctx.doubleTy()) : L;
      llvm::Value *rD = rIsI1 ? ctx.builder.CreateUIToFP(R, ctx.doubleTy()) : R;
      llvm::CmpInst::Predicate pred;
      if (op == "==")
        pred = llvm::CmpInst::FCMP_OEQ;
      else if (op == "!=")
        pred = llvm::CmpInst::FCMP_ONE;
      else if (op == "<")
        pred = llvm::CmpInst::FCMP_OLT;
      else if (op == ">")
        pred = llvm::CmpInst::FCMP_OGT;
      else if (op == "<=")
        pred = llvm::CmpInst::FCMP_OLE;
      else
        pred = llvm::CmpInst::FCMP_OGE;
      return ctx.builder.CreateFCmp(pred, lD, rD, "cmp");
    }
    if (op == "&&" || op == "||") {
      llvm::Value *lB =
          lIsI1 ? L
                : ctx.builder.CreateFCmpONE(
                      L, llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
      llvm::Value *rB =
          rIsI1 ? R
                : ctx.builder.CreateFCmpONE(
                      R, llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
      if (op == "&&")
        return ctx.builder.CreateAnd(lB, rB, "and");
      return ctx.builder.CreateOr(lB, rB, "or");
    }
    return nullptr;
  }
#endif
};

class UnaryExpr : public Expr {
  std::string op;
  std::unique_ptr<Expr> operand;

public:
  UnaryExpr(std::string o, std::unique_ptr<Expr> e)
      : op(std::move(o)), operand(std::move(e)) {}

  Value eval(std::shared_ptr<Env> env) override {
    Value val = operand->eval(env);
    if (op == "-") {
      // Keep as INT if operand is INT
      if (val.type == ValueType::INT)
        return Value(-val.toInt());
      return Value(-val.toNumber());
    }
    if (op == "!")
      return Value(!val.toBool());
    throw std::runtime_error("Unknown unary operator: " + op);
  }
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &ctx) override {
    llvm::Value *V = operand->codegen(ctx);
    if (!V)
      return nullptr;
    if (op == "-") {
      if (V->getType()->isIntegerTy(1))
        V = ctx.builder.CreateUIToFP(V, ctx.doubleTy());
      return ctx.builder.CreateFNeg(V, "neg");
    }
    if (op == "!") {
      if (!V->getType()->isIntegerTy(1))
        V = ctx.builder.CreateFCmpONE(
            V, llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
      return ctx.builder.CreateNot(V, "not");
    }
    return nullptr;
  }
#endif
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
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &ctx) override {
    llvm::Value *cond = condition->codegen(ctx);
    if (!cond)
      return nullptr;
    if (!cond->getType()->isIntegerTy(1))
      cond = ctx.builder.CreateFCmpONE(
          cond, llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
    llvm::Value *tVal = trueExpr->codegen(ctx);
    llvm::Value *fVal = falseExpr->codegen(ctx);
    if (!tVal || !fVal)
      return nullptr;
    return ctx.builder.CreateSelect(cond, tVal, fVal, "ternary");
  }
#endif
};

class CallExpr : public Expr {
  std::unique_ptr<Expr> callee;
  std::vector<std::unique_ptr<Expr>> args;

public:
  CallExpr(std::unique_ptr<Expr> c, std::vector<std::unique_ptr<Expr>> a)
      : callee(std::move(c)), args(std::move(a)) {}

  Value eval(std::shared_ptr<Env> env) override;
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &ctx) override {
    VarExpr *varCallee = dynamic_cast<VarExpr *>(callee.get());
    if (!varCallee)
      throw std::runtime_error(
          "Only named function calls supported in LLVM codegen");
    std::string fname = varCallee->getName();
    auto it = ctx.functions.find(fname);
    if (it == ctx.functions.end())
      throw std::runtime_error("Unknown function: " + fname);
    llvm::Function *F = it->second;
    std::vector<llvm::Value *> argVals;
    for (const auto &arg : args) {
      llvm::Value *a = arg->codegen(ctx);
      if (!a)
        return nullptr;
      if (a->getType()->isIntegerTy(1))
        a = ctx.builder.CreateUIToFP(a, ctx.doubleTy());
      argVals.push_back(a);
    }
    return ctx.builder.CreateCall(F, argVals, "call");
  }
#endif
};

class ClosureExpr : public Expr {
  std::vector<std::string> params;
  std::shared_ptr<Stmt> body; // Changed to shared_ptr

public:
  ClosureExpr(std::vector<std::string> p, std::unique_ptr<Stmt> b)
      : params(std::move(p)), body(std::move(b)) {}

  Value eval(std::shared_ptr<Env> env) override {
    // Capture current environment for closure and share the body
    auto func = std::make_shared<Function>(params, body, env);
    return Value(func);
  }
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &) override {
    throw std::runtime_error(
        "Anonymous function (closure) not supported in LLVM codegen");
    return nullptr;
  }
#endif
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
#ifdef USE_LLVM_CODEGEN
  llvm::Value *codegen(CodeGenContext &) override { return nullptr; }
#endif
};

// ===== Control Flow Exceptions =====
struct ReturnException {
  Value value;
  ReturnException(Value v) : value(v) {}
};

// NEW: Break and Continue exceptions for loop control
struct BreakException {};
struct ContinueException {};

// ===== Statement AST =====
class Stmt {
public:
  virtual ~Stmt() = default;
  virtual void exec(std::shared_ptr<Env> env) = 0;
#ifdef USE_LLVM_CODEGEN
  virtual void codegen(CodeGenContext &ctx) {}
#endif
};

class ExprStmt : public Stmt {
  std::unique_ptr<Expr> expr;

public:
  ExprStmt(std::unique_ptr<Expr> e) : expr(std::move(e)) {}
  void exec(std::shared_ptr<Env> env) override { expr->eval(env); }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    if (expr)
      expr->codegen(ctx);
  }
#endif
};

class BlockStmt : public Stmt {
public:
  std::vector<std::unique_ptr<Stmt>> statements;

  void exec(std::shared_ptr<Env> env) override {
    auto blockEnv = std::make_shared<Env>(env);
    for (auto &stmt : statements)
      stmt->exec(blockEnv);
  }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    ctx.pushScope();
    for (auto &stmt : statements)
      stmt->codegen(ctx);
    ctx.popScope();
  }
#endif
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
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    llvm::AllocaInst *alloca =
        ctx.builder.CreateAlloca(ctx.doubleTy(), nullptr, name);
    ctx.defineAlloca(name, alloca);
    if (init) {
      llvm::Value *v = init->codegen(ctx);
      if (v) {
        if (v->getType()->isIntegerTy(1))
          v = ctx.builder.CreateUIToFP(v, ctx.doubleTy());
        ctx.builder.CreateStore(v, alloca);
      }
    } else {
      ctx.builder.CreateStore(llvm::ConstantFP::get(ctx.doubleTy(), 0.0),
                              alloca);
    }
  }
#endif
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
        else if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          env->set(name, Value(oldVal.toInt() + newVal.toInt()));
        else
          env->set(name, Value(oldVal.toNumber() + newVal.toNumber()));
      } else if (op == "-=") {
        if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          env->set(name, Value(oldVal.toInt() - newVal.toInt()));
        else
          env->set(name, Value(oldVal.toNumber() - newVal.toNumber()));
      } else if (op == "*=") {
        if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          env->set(name, Value(oldVal.toInt() * newVal.toInt()));
        else
          env->set(name, Value(oldVal.toNumber() * newVal.toNumber()));
      } else if (op == "/=") {
        if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          env->set(name, Value(oldVal.toInt() / newVal.toInt()));
        else
          env->set(name, Value(oldVal.toNumber() / newVal.toNumber()));
      }
    }
  }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    llvm::AllocaInst *alloca = ctx.findAlloca(name);
    if (!alloca)
      throw std::runtime_error("Unknown variable in assignment: " + name);
    llvm::Value *newVal = value->codegen(ctx);
    if (!newVal)
      return;
    if (newVal->getType()->isIntegerTy(1))
      newVal = ctx.builder.CreateUIToFP(newVal, ctx.doubleTy());
    if (op == "=") {
      ctx.builder.CreateStore(newVal, alloca);
      return;
    }
    llvm::Value *oldVal = ctx.builder.CreateLoad(ctx.doubleTy(), alloca, name);
    llvm::Value *result = nullptr;
    if (op == "+=")
      result = ctx.builder.CreateFAdd(oldVal, newVal, "add_assign");
    else if (op == "-=")
      result = ctx.builder.CreateFSub(oldVal, newVal, "sub_assign");
    else if (op == "*=")
      result = ctx.builder.CreateFMul(oldVal, newVal, "mul_assign");
    else if (op == "/=")
      result = ctx.builder.CreateFDiv(oldVal, newVal, "div_assign");
    if (result)
      ctx.builder.CreateStore(result, alloca);
  }
#endif
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
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    llvm::Function *printfFunc = ctx.module->getFunction("printf");
    if (!printfFunc) {
      std::vector<llvm::Type *> printfArgs;
      printfArgs.push_back(ctx.i8PtrTy());
      llvm::FunctionType *printfType =
          llvm::FunctionType::get(ctx.i32Ty(), printfArgs, true);
      printfFunc = llvm::Function::Create(
          printfType, llvm::Function::ExternalLinkage, "printf", ctx.module);
    }
    for (size_t i = 0; i < exprs.size(); i++) {
      llvm::Value *v = exprs[i]->codegen(ctx);
      if (!v)
        continue;
      if (v->getType()->isIntegerTy(1))
        v = ctx.builder.CreateUIToFP(v, ctx.doubleTy());
      llvm::Value *fmt = ctx.builder.CreateGlobalStringPtr(
          i < exprs.size() - 1 ? "%g " : "%g\n");
      ctx.builder.CreateCall(printfFunc, {fmt, v});
    }
  }
#endif
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
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    llvm::Value *cond = condition->codegen(ctx);
    if (!cond)
      return;
    if (!cond->getType()->isIntegerTy(1))
      cond = ctx.builder.CreateFCmpONE(
          cond, llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
    llvm::Function *fn = ctx.builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBB =
        llvm::BasicBlock::Create(ctx.context, "if.then", fn);
    llvm::BasicBlock *elseBB =
        elseStmt ? llvm::BasicBlock::Create(ctx.context, "if.else", fn)
                 : nullptr;
    llvm::BasicBlock *mergeBB =
        llvm::BasicBlock::Create(ctx.context, "if.end", fn);
    if (elseBB)
      ctx.builder.CreateCondBr(cond, thenBB, elseBB);
    else
      ctx.builder.CreateCondBr(cond, thenBB, mergeBB);
    ctx.builder.SetInsertPoint(thenBB);
    thenStmt->codegen(ctx);
    if (!ctx.builder.GetInsertBlock()->getTerminator())
      ctx.builder.CreateBr(mergeBB);
    if (elseBB) {
      ctx.builder.SetInsertPoint(elseBB);
      elseStmt->codegen(ctx);
      if (!ctx.builder.GetInsertBlock()->getTerminator())
        ctx.builder.CreateBr(mergeBB);
    }
    ctx.builder.SetInsertPoint(mergeBB);
  }
#endif
};

// NEW: Enhanced WhileStmt that handles break and continue
class WhileStmt : public Stmt {
  std::unique_ptr<Expr> condition;
  std::unique_ptr<Stmt> body;

public:
  WhileStmt(std::unique_ptr<Expr> c, std::unique_ptr<Stmt> b)
      : condition(std::move(c)), body(std::move(b)) {}

  void exec(std::shared_ptr<Env> env) override {
    try {
      while (condition->eval(env).toBool()) {
        try {
          body->exec(env);
        } catch (ContinueException &) {
          // Continue to next iteration
          continue;
        }
      }
    } catch (BreakException &) {
      // Break out of the loop
    }
  }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    llvm::Function *fn = ctx.builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *condBB =
        llvm::BasicBlock::Create(ctx.context, "while.cond", fn);
    llvm::BasicBlock *bodyBB =
        llvm::BasicBlock::Create(ctx.context, "while.body", fn);
    llvm::BasicBlock *exitBB =
        llvm::BasicBlock::Create(ctx.context, "while.exit", fn);
    ctx.builder.CreateBr(condBB);
    ctx.builder.SetInsertPoint(condBB);
    ctx.loopStack.push_back({condBB, exitBB});
    llvm::Value *cond = condition->codegen(ctx);
    if (!cond)
      cond = llvm::ConstantInt::get(ctx.i1Ty(), 1);
    else if (!cond->getType()->isIntegerTy(1))
      cond = ctx.builder.CreateFCmpONE(
          cond, llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
    ctx.builder.CreateCondBr(cond, bodyBB, exitBB);
    ctx.builder.SetInsertPoint(bodyBB);
    body->codegen(ctx);
    if (!ctx.builder.GetInsertBlock()->getTerminator())
      ctx.builder.CreateBr(condBB);
    ctx.loopStack.pop_back();
    ctx.builder.SetInsertPoint(exitBB);
  }
#endif
};

class ReturnStmt : public Stmt {
  std::unique_ptr<Expr> value;

public:
  ReturnStmt(std::unique_ptr<Expr> v = nullptr) : value(std::move(v)) {}

  void exec(std::shared_ptr<Env> env) override {
    Value val = value ? value->eval(env) : Value();
    throw ReturnException(val);
  }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    if (value) {
      llvm::Value *v = value->codegen(ctx);
      if (v) {
        if (v->getType()->isIntegerTy(1))
          v = ctx.builder.CreateUIToFP(v, ctx.doubleTy());
        ctx.builder.CreateRet(v);
      } else {
        ctx.builder.CreateRet(llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
      }
    } else {
      llvm::Type *retTy = ctx.currentFunction
                              ? ctx.currentFunction->getReturnType()
                              : llvm::Type::getInt32Ty(ctx.context);
      if (retTy->isIntegerTy())
        ctx.builder.CreateRet(llvm::ConstantInt::get(retTy, 0));
      else
        ctx.builder.CreateRet(llvm::ConstantFP::get(retTy, 0.0));
    }
  }
#endif
};

// NEW: Break statement
class BreakStmt : public Stmt {
public:
  void exec(std::shared_ptr<Env>) override { throw BreakException(); }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    if (ctx.loopStack.empty())
      throw std::runtime_error("break outside loop");
    ctx.builder.CreateBr(ctx.loopStack.back().exit);
  }
#endif
};

// NEW: Continue statement
class ContinueStmt : public Stmt {
public:
  void exec(std::shared_ptr<Env>) override { throw ContinueException(); }
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    if (ctx.loopStack.empty())
      throw std::runtime_error("continue outside loop");
    ctx.builder.CreateBr(ctx.loopStack.back().cond);
  }
#endif
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
#ifdef USE_LLVM_CODEGEN
  void codegen(CodeGenContext &ctx) override {
    std::vector<llvm::Type *> paramTypes(params.size(), ctx.doubleTy());
    llvm::FunctionType *FT =
        llvm::FunctionType::get(ctx.doubleTy(), paramTypes, false);
    llvm::Function *F = llvm::Function::Create(
        FT, llvm::Function::InternalLinkage, name, ctx.module);
    ctx.functions[name] = F;
    unsigned idx = 0;
    for (auto &arg : F->args())
      arg.setName(params[idx++]);
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(ctx.context, "entry", F);
    ctx.builder.SetInsertPoint(BB);
    ctx.pushScope();
    idx = 0;
    for (auto &arg : F->args()) {
      llvm::AllocaInst *alloca =
          ctx.builder.CreateAlloca(ctx.doubleTy(), nullptr, arg.getName());
      ctx.builder.CreateStore(&arg, alloca);
      ctx.defineAlloca(std::string(arg.getName()), alloca);
    }
    ctx.currentFunction = F;
    body->codegen(ctx);
    ctx.currentFunction = nullptr;
    if (!ctx.builder.GetInsertBlock()->getTerminator())
      ctx.builder.CreateRet(llvm::ConstantFP::get(ctx.doubleTy(), 0.0));
    ctx.popScope();
  }
#endif
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
  } catch (ReturnException &retEx) {
    return retEx.value;
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

    // Anonymous function expression: fn ( params ) { body }
    if (curTok.type == TOK_FN) {
      advance(); // consume 'fn'
      expect(TOK_LP, "Expected '(' after 'fn'");
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
      return std::make_unique<ClosureExpr>(params, std::move(body));
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
    if (curTok.type == TOK_BREAK)
      return parseBreakStmt();
    if (curTok.type == TOK_CONTINUE)
      return parseContinueStmt();
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

  std::unique_ptr<Stmt> parseBreakStmt() {
    advance(); // consume 'break'
    expect(TOK_SEMI, "Expected ';' after break statement");
    return std::make_unique<BreakStmt>();
  }

  std::unique_ptr<Stmt> parseContinueStmt() {
    advance(); // consume 'continue'
    expect(TOK_SEMI, "Expected ';' after continue statement");
    return std::make_unique<ContinueStmt>();
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

#ifdef USE_LLVM_CODEGEN
// ===== Top-level IR generation: AST -> LLVM Module =====
inline void generateProgram(llvm::Module *module, llvm::LLVMContext &context,
                            llvm::IRBuilder<> &builder,
                            const std::vector<std::unique_ptr<Stmt>> &program) {
  CodeGenContext ctx{builder, module, context};
  ctx.pushScope();

  llvm::FunctionType *mainFT =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
  llvm::Function *mainFunc = llvm::Function::Create(
      mainFT, llvm::Function::ExternalLinkage, "main", module);
  llvm::BasicBlock *mainEntry =
      llvm::BasicBlock::Create(context, "entry", mainFunc);
  builder.SetInsertPoint(mainEntry);

  for (const auto &stmt : program) {
    if (dynamic_cast<FunctionStmt *>(stmt.get())) {
      llvm::BasicBlock *savedBlock = builder.GetInsertBlock();
      stmt->codegen(ctx);
      builder.SetInsertPoint(savedBlock);
    }
  }
  builder.SetInsertPoint(mainEntry);
  for (const auto &stmt : program) {
    if (!dynamic_cast<FunctionStmt *>(stmt.get()))
      stmt->codegen(ctx);
  }

  llvm::BasicBlock *cur = builder.GetInsertBlock();
  if (cur && !cur->getTerminator())
    builder.CreateRet(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0));
  ctx.popScope();
}
#endif

} // namespace Interpreter_V4
