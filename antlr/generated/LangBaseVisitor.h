
// Generated from Lang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "LangVisitor.h"


/**
 * This class provides an empty implementation of LangVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  LangBaseVisitor : public LangVisitor {
public:

  virtual std::any visitProgram(LangParser::ProgramContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarDeclStmt(LangParser::VarDeclStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFuncDeclStmt(LangParser::FuncDeclStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfStmt(LangParser::IfStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitWhileStmt(LangParser::WhileStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPrintStmt(LangParser::PrintStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReturnStmt(LangParser::ReturnStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBreakStmt(LangParser::BreakStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitContinueStmt(LangParser::ContinueStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlockStmt(LangParser::BlockStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAssignStmt(LangParser::AssignStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExprStmt(LangParser::ExprStmtContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBlock(LangParser::BlockContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarDecl(LangParser::VarDeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionDecl(LangParser::FunctionDeclContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfStatement(LangParser::IfStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitWhileStatement(LangParser::WhileStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPrintStatement(LangParser::PrintStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReturnStatement(LangParser::ReturnStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAssignmentStatement(LangParser::AssignmentStatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameterList(LangParser::ParameterListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpressionList(LangParser::ExpressionListContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpression(LangParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTernaryExpression(LangParser::TernaryExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalOr(LangParser::LogicalOrContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalAnd(LangParser::LogicalAndContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEquality(LangParser::EqualityContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRelational(LangParser::RelationalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAdditive(LangParser::AdditiveContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMultiplicative(LangParser::MultiplicativeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryOp(LangParser::UnaryOpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTypeofOp(LangParser::TypeofOpContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPostfixExpr(LangParser::PostfixExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionCall(LangParser::FunctionCallContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntLiteral(LangParser::IntLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDoubleLiteral(LangParser::DoubleLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringLiteral(LangParser::StringLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBoolTrue(LangParser::BoolTrueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBoolFalse(LangParser::BoolFalseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVariable(LangParser::VariableContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenthesized(LangParser::ParenthesizedContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnonymousFunction(LangParser::AnonymousFunctionContext *ctx) override {
    return visitChildren(ctx);
  }


};

