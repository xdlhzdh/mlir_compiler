#pragma once

#include <any>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "generated/LangBaseVisitor.h"
#include "generated/LangParser.h"

namespace LangInterpreter {

// ===== Type System =====
enum class ValueType { NONE, INT, DOUBLE, STRING, BOOL, FUNCTION };

struct Value;
struct Env;
struct Function;

// Function type
struct Function {
  std::vector<std::string> params;
  LangParser::BlockStatementContext *body;
  std::shared_ptr<Env> capturedEnv;

  Function(std::vector<std::string> p, LangParser::BlockStatementContext *b,
           std::shared_ptr<Env> env)
      : params(std::move(p)), body(b), capturedEnv(env) {}
};

struct Value {
  ValueType type;
  std::variant<int, double, std::string, bool, std::shared_ptr<Function>> data;

  Value() : type(ValueType::NONE), data(0) {}
  Value(int v) : type(ValueType::INT), data(v) {}
  Value(double v) : type(ValueType::DOUBLE), data(v) {}
  Value(const std::string &v) : type(ValueType::STRING), data(v) {}
  Value(bool v) : type(ValueType::BOOL), data(v) {}
  Value(std::shared_ptr<Function> v) : type(ValueType::FUNCTION), data(v) {}

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

// ===== Environment =====
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

// ===== Control Flow Exceptions =====
struct ReturnException {
  Value value;
  ReturnException(Value v) : value(v) {}
};

struct BreakException {};
struct ContinueException {};

// ===== Interpreter Visitor =====
class InterpreterVisitor : public LangBaseVisitor {
private:
  std::shared_ptr<Env> currentEnv;

public:
  InterpreterVisitor() : currentEnv(std::make_shared<Env>()) {}

  // Helper to extract Value from std::any
  Value toValue(std::any result) {
    if (!result.has_value())
      return Value();
    try {
      return std::any_cast<Value>(result);
    } catch (const std::bad_any_cast &) {
      return Value();
    }
  }

  // Program
  std::any visitProgram(LangParser::ProgramContext *ctx) override {
    for (auto stmt : ctx->statement()) {
      visit(stmt);
    }
    return std::any();
  }

  // Statements
  std::any visitVarDeclStmt(LangParser::VarDeclStmtContext *ctx) override {
    auto varDecl = ctx->varDecl();
    std::string name = varDecl->ID()->getText();
    Value value = toValue(visit(varDecl->expression()));
    currentEnv->define(name, value);
    return std::any();
  }

  std::any visitAssignStmt(LangParser::AssignStmtContext *ctx) override {
    auto assign = ctx->assignmentStatement();
    std::string name = assign->ID()->getText();
    Value newVal = toValue(visit(assign->expression()));

    std::string op = "=";
    if (assign->ADD_ASSIGN())
      op = "+=";
    else if (assign->SUB_ASSIGN())
      op = "-=";
    else if (assign->MUL_ASSIGN())
      op = "*=";
    else if (assign->DIV_ASSIGN())
      op = "/=";

    if (op == "=") {
      currentEnv->set(name, newVal);
    } else {
      Value oldVal = currentEnv->get(name);
      if (op == "+=") {
        if (oldVal.type == ValueType::STRING ||
            newVal.type == ValueType::STRING)
          currentEnv->set(name, Value(oldVal.toString() + newVal.toString()));
        else if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          currentEnv->set(name, Value(oldVal.toInt() + newVal.toInt()));
        else
          currentEnv->set(name, Value(oldVal.toNumber() + newVal.toNumber()));
      } else if (op == "-=") {
        if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          currentEnv->set(name, Value(oldVal.toInt() - newVal.toInt()));
        else
          currentEnv->set(name, Value(oldVal.toNumber() - newVal.toNumber()));
      } else if (op == "*=") {
        if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          currentEnv->set(name, Value(oldVal.toInt() * newVal.toInt()));
        else
          currentEnv->set(name, Value(oldVal.toNumber() * newVal.toNumber()));
      } else if (op == "/=") {
        if (oldVal.type == ValueType::INT && newVal.type == ValueType::INT)
          currentEnv->set(name, Value(oldVal.toInt() / newVal.toInt()));
        else
          currentEnv->set(name, Value(oldVal.toNumber() / newVal.toNumber()));
      }
    }
    return std::any();
  }

  std::any visitPrintStmt(LangParser::PrintStmtContext *ctx) override {
    auto printStmt = ctx->printStatement();
    auto exprs = printStmt->expressionList()->expression();
    for (size_t i = 0; i < exprs.size(); i++) {
      Value val = toValue(visit(exprs[i]));
      std::cout << val.toString();
      if (i < exprs.size() - 1)
        std::cout << " ";
    }
    std::cout << std::endl;
    return std::any();
  }

  std::any visitReturnStmt(LangParser::ReturnStmtContext *ctx) override {
    auto returnStmt = ctx->returnStatement();
    Value val;
    if (returnStmt->expression())
      val = toValue(visit(returnStmt->expression()));
    throw ReturnException(val);
  }

  std::any visitBreakStmt(LangParser::BreakStmtContext *ctx) override {
    throw BreakException();
  }

  std::any visitContinueStmt(LangParser::ContinueStmtContext *ctx) override {
    throw ContinueException();
  }

  std::any visitIfStmt(LangParser::IfStmtContext *ctx) override {
    auto ifStmt = ctx->ifStatement();
    Value condition = toValue(visit(ifStmt->expression()));
    if (condition.toBool()) {
      visit(ifStmt->statement(0));
    } else if (ifStmt->statement().size() > 1) {
      visit(ifStmt->statement(1));
    }
    return std::any();
  }

  std::any visitWhileStmt(LangParser::WhileStmtContext *ctx) override {
    auto whileStmt = ctx->whileStatement();
    try {
      while (toValue(visit(whileStmt->expression())).toBool()) {
        try {
          visit(whileStmt->statement());
        } catch (ContinueException &) {
          continue;
        }
      }
    } catch (BreakException &) {
    }
    return std::any();
  }

  std::any visitBlockStmt(LangParser::BlockStmtContext *ctx) override {
    return visit(ctx->blockStatement());
  }

  std::any visitBlock(LangParser::BlockContext *ctx) override {
    auto savedEnv = currentEnv;
    currentEnv = std::make_shared<Env>(currentEnv);
    for (auto stmt : ctx->statement()) {
      visit(stmt);
    }
    currentEnv = savedEnv;
    return std::any();
  }

  std::any visitExprStmt(LangParser::ExprStmtContext *ctx) override {
    visit(ctx->expression());
    return std::any();
  }

  std::any visitFuncDeclStmt(LangParser::FuncDeclStmtContext *ctx) override {
    auto funcDecl = ctx->functionDecl();
    std::string name = funcDecl->ID()->getText();

    std::vector<std::string> params;
    if (funcDecl->parameterList()) {
      for (auto id : funcDecl->parameterList()->ID()) {
        params.push_back(id->getText());
      }
    }

    auto func = std::make_shared<Function>(params, funcDecl->blockStatement(),
                                           currentEnv);
    currentEnv->define(name, Value(func));
    return std::any();
  }

  // Expressions
  std::any visitExpression(LangParser::ExpressionContext *ctx) override {
    return visit(ctx->ternaryExpression());
  }

  std::any visitIntLiteral(LangParser::IntLiteralContext *ctx) override {
    return Value(std::stoi(ctx->INT()->getText()));
  }

  std::any visitDoubleLiteral(LangParser::DoubleLiteralContext *ctx) override {
    return Value(std::stod(ctx->DOUBLE()->getText()));
  }

  std::any visitStringLiteral(LangParser::StringLiteralContext *ctx) override {
    std::string str = ctx->STRING()->getText();
    // Remove quotes
    return Value(str.substr(1, str.length() - 2));
  }

  std::any visitBoolTrue(LangParser::BoolTrueContext *ctx) override {
    return Value(true);
  }

  std::any visitBoolFalse(LangParser::BoolFalseContext *ctx) override {
    return Value(false);
  }

  std::any visitVariable(LangParser::VariableContext *ctx) override {
    return currentEnv->get(ctx->ID()->getText());
  }

  std::any visitParenthesized(LangParser::ParenthesizedContext *ctx) override {
    return visit(ctx->expression());
  }

  std::any
  visitAnonymousFunction(LangParser::AnonymousFunctionContext *ctx) override {
    std::vector<std::string> params;
    if (ctx->parameterList()) {
      for (auto id : ctx->parameterList()->ID()) {
        params.push_back(id->getText());
      }
    }

    auto func =
        std::make_shared<Function>(params, ctx->blockStatement(), currentEnv);
    return Value(func);
  }

  std::any visitPostfixExpr(LangParser::PostfixExprContext *ctx) override {
    return visit(ctx->postfix());
  }

  std::any visitFunctionCall(LangParser::FunctionCallContext *ctx) override {
    Value result = toValue(visit(ctx->primary()));

    // Check if primary is a variable for postfix decrement
    std::string varName;
    if (auto varCtx =
            dynamic_cast<LangParser::VariableContext *>(ctx->primary())) {
      varName = varCtx->ID()->getText();
    }

    // Process all postfix operations (function calls and decrements)
    auto exprLists = ctx->expressionList();
    auto decTokens = ctx->DEC();

    size_t exprIdx = 0;
    size_t decIdx = 0;

    // Iterate through all children to process in order
    for (size_t i = 1; i < ctx->children.size(); i++) {
      auto child = ctx->children[i];

      // Check if this is a DEC token
      if (child->getText() == "--") {
        // Postfix decrement: x-- (returns old value, then decrements)
        if (varName.empty()) {
          throw std::runtime_error("Postfix decrement requires a variable");
        }
        Value oldVal = currentEnv->get(varName);
        Value newVal;
        if (oldVal.type == ValueType::INT)
          newVal = Value(oldVal.toInt() - 1);
        else
          newVal = Value(oldVal.toNumber() - 1.0);
        currentEnv->set(varName, newVal);
        result = oldVal; // Return the old value
        decIdx++;
      }
      // Check if this is a function call (LPAREN)
      else if (child->getText() == "(") {
        if (result.type != ValueType::FUNCTION)
          throw std::runtime_error("Not a function");

        auto func = std::get<std::shared_ptr<Function>>(result.data);

        std::vector<Value> args;
        if (exprIdx < exprLists.size()) {
          for (auto expr : exprLists[exprIdx]->expression()) {
            args.push_back(toValue(visit(expr)));
          }
          exprIdx++;
        }

        if (args.size() != func->params.size())
          throw std::runtime_error("Argument count mismatch");

        auto savedEnv = currentEnv;
        currentEnv = std::make_shared<Env>(func->capturedEnv);
        for (size_t j = 0; j < args.size(); j++) {
          currentEnv->define(func->params[j], args[j]);
        }

        try {
          visit(func->body);
          currentEnv = savedEnv;
          result = Value();
        } catch (ReturnException &e) {
          currentEnv = savedEnv;
          result = e.value;
        }
      }
    }

    return result;
  }

  std::any visitUnaryOp(LangParser::UnaryOpContext *ctx) override {
    if (ctx->DEC()) {
      // Prefix decrement: --x
      // We need to recursively find the variable to decrement
      auto innerUnary = ctx->unary();

      // Navigate through the unary chain to find the variable
      std::function<std::string(LangParser::UnaryContext *)> getVarName;
      getVarName = [&](LangParser::UnaryContext *unaryCtx) -> std::string {
        // Try to cast to PostfixExprContext
        if (auto postfixCtx =
                dynamic_cast<LangParser::PostfixExprContext *>(unaryCtx)) {
          auto postfix = postfixCtx->postfix();
          // Cast postfix to FunctionCallContext
          if (auto funcCallCtx =
                  dynamic_cast<LangParser::FunctionCallContext *>(postfix)) {
            // Check if it's a simple variable (no function calls)
            if (funcCallCtx->expressionList().empty()) {
              if (auto varCtx = dynamic_cast<LangParser::VariableContext *>(
                      funcCallCtx->primary())) {
                return varCtx->ID()->getText();
              }
            }
          }
        }
        return "";
      };

      std::string varName = getVarName(innerUnary);
      if (varName.empty()) {
        throw std::runtime_error("Decrement operator requires a variable");
      }

      Value oldVal = currentEnv->get(varName);
      Value newVal;
      if (oldVal.type == ValueType::INT)
        newVal = Value(oldVal.toInt() - 1);
      else
        newVal = Value(oldVal.toNumber() - 1.0);
      currentEnv->set(varName, newVal);
      return newVal;
    }

    Value val = toValue(visit(ctx->unary()));
    if (ctx->MINUS()) {
      if (val.type == ValueType::INT)
        return Value(-val.toInt());
      else
        return Value(-val.toNumber());
    }
    if (ctx->NOT())
      return Value(!val.toBool());
    return val;
  }

  std::any visitTypeofOp(LangParser::TypeofOpContext *ctx) override {
    Value val = toValue(visit(ctx->unary()));
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

  std::any visitAdditive(LangParser::AdditiveContext *ctx) override {
    Value left = toValue(visit(ctx->multiplicative(0)));
    for (size_t i = 1; i < ctx->multiplicative().size(); i++) {
      Value right = toValue(visit(ctx->multiplicative(i)));
      if (ctx->PLUS(i - 1)) {
        if (left.type == ValueType::STRING || right.type == ValueType::STRING)
          left = Value(left.toString() + right.toString());
        else if (left.type == ValueType::INT && right.type == ValueType::INT)
          left = Value(left.toInt() + right.toInt());
        else
          left = Value(left.toNumber() + right.toNumber());
      } else if (ctx->MINUS(i - 1)) {
        if (left.type == ValueType::INT && right.type == ValueType::INT)
          left = Value(left.toInt() - right.toInt());
        else
          left = Value(left.toNumber() - right.toNumber());
      }
    }
    return left;
  }

  std::any
  visitMultiplicative(LangParser::MultiplicativeContext *ctx) override {
    Value left = toValue(visit(ctx->unary(0)));
    for (size_t i = 1; i < ctx->unary().size(); i++) {
      Value right = toValue(visit(ctx->unary(i)));
      if (ctx->MUL(i - 1)) {
        if (left.type == ValueType::INT && right.type == ValueType::INT)
          left = Value(left.toInt() * right.toInt());
        else
          left = Value(left.toNumber() * right.toNumber());
      } else if (ctx->DIV(i - 1)) {
        if (left.type == ValueType::INT && right.type == ValueType::INT)
          left = Value(left.toInt() / right.toInt());
        else
          left = Value(left.toNumber() / right.toNumber());
      } else if (ctx->MOD(i - 1))
        left = Value(left.toInt() % right.toInt());
    }
    return left;
  }

  std::any visitRelational(LangParser::RelationalContext *ctx) override {
    Value left = toValue(visit(ctx->additive(0)));
    for (size_t i = 1; i < ctx->additive().size(); i++) {
      Value right = toValue(visit(ctx->additive(i)));
      if (ctx->LT(i - 1))
        left = Value(left.toNumber() < right.toNumber());
      else if (ctx->GT(i - 1))
        left = Value(left.toNumber() > right.toNumber());
      else if (ctx->LE(i - 1))
        left = Value(left.toNumber() <= right.toNumber());
      else if (ctx->GE(i - 1))
        left = Value(left.toNumber() >= right.toNumber());
    }
    return left;
  }

  std::any visitEquality(LangParser::EqualityContext *ctx) override {
    Value left = toValue(visit(ctx->relational(0)));
    for (size_t i = 1; i < ctx->relational().size(); i++) {
      Value right = toValue(visit(ctx->relational(i)));
      if (ctx->EQ(i - 1)) {
        if (left.type == ValueType::STRING && right.type == ValueType::STRING)
          left = Value(left.toString() == right.toString());
        else
          left = Value(left.toNumber() == right.toNumber());
      } else if (ctx->NEQ(i - 1)) {
        if (left.type == ValueType::STRING && right.type == ValueType::STRING)
          left = Value(left.toString() != right.toString());
        else
          left = Value(left.toNumber() != right.toNumber());
      }
    }
    return left;
  }

  std::any visitLogicalAnd(LangParser::LogicalAndContext *ctx) override {
    Value left = toValue(visit(ctx->equality(0)));
    for (size_t i = 1; i < ctx->equality().size(); i++) {
      if (!left.toBool())
        return left;
      left = toValue(visit(ctx->equality(i)));
    }
    return left;
  }

  std::any visitLogicalOr(LangParser::LogicalOrContext *ctx) override {
    Value left = toValue(visit(ctx->logicalAnd(0)));
    for (size_t i = 1; i < ctx->logicalAnd().size(); i++) {
      if (left.toBool())
        return left;
      left = toValue(visit(ctx->logicalAnd(i)));
    }
    return left;
  }

  std::any
  visitTernaryExpression(LangParser::TernaryExpressionContext *ctx) override {
    Value condition = toValue(visit(ctx->logicalOr()));
    if (ctx->QUESTION()) {
      if (condition.toBool())
        return visit(ctx->expression(0));
      else
        return visit(ctx->expression(1));
    }
    return condition;
  }
};

} // namespace LangInterpreter
