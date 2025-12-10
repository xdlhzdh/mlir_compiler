
// Generated from Lang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "LangParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by LangParser.
 */
class  LangListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterProgram(LangParser::ProgramContext *ctx) = 0;
  virtual void exitProgram(LangParser::ProgramContext *ctx) = 0;

  virtual void enterVarDeclStmt(LangParser::VarDeclStmtContext *ctx) = 0;
  virtual void exitVarDeclStmt(LangParser::VarDeclStmtContext *ctx) = 0;

  virtual void enterFuncDeclStmt(LangParser::FuncDeclStmtContext *ctx) = 0;
  virtual void exitFuncDeclStmt(LangParser::FuncDeclStmtContext *ctx) = 0;

  virtual void enterIfStmt(LangParser::IfStmtContext *ctx) = 0;
  virtual void exitIfStmt(LangParser::IfStmtContext *ctx) = 0;

  virtual void enterWhileStmt(LangParser::WhileStmtContext *ctx) = 0;
  virtual void exitWhileStmt(LangParser::WhileStmtContext *ctx) = 0;

  virtual void enterPrintStmt(LangParser::PrintStmtContext *ctx) = 0;
  virtual void exitPrintStmt(LangParser::PrintStmtContext *ctx) = 0;

  virtual void enterReturnStmt(LangParser::ReturnStmtContext *ctx) = 0;
  virtual void exitReturnStmt(LangParser::ReturnStmtContext *ctx) = 0;

  virtual void enterBreakStmt(LangParser::BreakStmtContext *ctx) = 0;
  virtual void exitBreakStmt(LangParser::BreakStmtContext *ctx) = 0;

  virtual void enterContinueStmt(LangParser::ContinueStmtContext *ctx) = 0;
  virtual void exitContinueStmt(LangParser::ContinueStmtContext *ctx) = 0;

  virtual void enterBlockStmt(LangParser::BlockStmtContext *ctx) = 0;
  virtual void exitBlockStmt(LangParser::BlockStmtContext *ctx) = 0;

  virtual void enterAssignStmt(LangParser::AssignStmtContext *ctx) = 0;
  virtual void exitAssignStmt(LangParser::AssignStmtContext *ctx) = 0;

  virtual void enterExprStmt(LangParser::ExprStmtContext *ctx) = 0;
  virtual void exitExprStmt(LangParser::ExprStmtContext *ctx) = 0;

  virtual void enterBlock(LangParser::BlockContext *ctx) = 0;
  virtual void exitBlock(LangParser::BlockContext *ctx) = 0;

  virtual void enterVarDecl(LangParser::VarDeclContext *ctx) = 0;
  virtual void exitVarDecl(LangParser::VarDeclContext *ctx) = 0;

  virtual void enterFunctionDecl(LangParser::FunctionDeclContext *ctx) = 0;
  virtual void exitFunctionDecl(LangParser::FunctionDeclContext *ctx) = 0;

  virtual void enterIfStatement(LangParser::IfStatementContext *ctx) = 0;
  virtual void exitIfStatement(LangParser::IfStatementContext *ctx) = 0;

  virtual void enterWhileStatement(LangParser::WhileStatementContext *ctx) = 0;
  virtual void exitWhileStatement(LangParser::WhileStatementContext *ctx) = 0;

  virtual void enterPrintStatement(LangParser::PrintStatementContext *ctx) = 0;
  virtual void exitPrintStatement(LangParser::PrintStatementContext *ctx) = 0;

  virtual void enterReturnStatement(LangParser::ReturnStatementContext *ctx) = 0;
  virtual void exitReturnStatement(LangParser::ReturnStatementContext *ctx) = 0;

  virtual void enterAssignmentStatement(LangParser::AssignmentStatementContext *ctx) = 0;
  virtual void exitAssignmentStatement(LangParser::AssignmentStatementContext *ctx) = 0;

  virtual void enterParameterList(LangParser::ParameterListContext *ctx) = 0;
  virtual void exitParameterList(LangParser::ParameterListContext *ctx) = 0;

  virtual void enterExpressionList(LangParser::ExpressionListContext *ctx) = 0;
  virtual void exitExpressionList(LangParser::ExpressionListContext *ctx) = 0;

  virtual void enterExpression(LangParser::ExpressionContext *ctx) = 0;
  virtual void exitExpression(LangParser::ExpressionContext *ctx) = 0;

  virtual void enterTernaryExpression(LangParser::TernaryExpressionContext *ctx) = 0;
  virtual void exitTernaryExpression(LangParser::TernaryExpressionContext *ctx) = 0;

  virtual void enterLogicalOr(LangParser::LogicalOrContext *ctx) = 0;
  virtual void exitLogicalOr(LangParser::LogicalOrContext *ctx) = 0;

  virtual void enterLogicalAnd(LangParser::LogicalAndContext *ctx) = 0;
  virtual void exitLogicalAnd(LangParser::LogicalAndContext *ctx) = 0;

  virtual void enterEquality(LangParser::EqualityContext *ctx) = 0;
  virtual void exitEquality(LangParser::EqualityContext *ctx) = 0;

  virtual void enterRelational(LangParser::RelationalContext *ctx) = 0;
  virtual void exitRelational(LangParser::RelationalContext *ctx) = 0;

  virtual void enterAdditive(LangParser::AdditiveContext *ctx) = 0;
  virtual void exitAdditive(LangParser::AdditiveContext *ctx) = 0;

  virtual void enterMultiplicative(LangParser::MultiplicativeContext *ctx) = 0;
  virtual void exitMultiplicative(LangParser::MultiplicativeContext *ctx) = 0;

  virtual void enterUnaryOp(LangParser::UnaryOpContext *ctx) = 0;
  virtual void exitUnaryOp(LangParser::UnaryOpContext *ctx) = 0;

  virtual void enterTypeofOp(LangParser::TypeofOpContext *ctx) = 0;
  virtual void exitTypeofOp(LangParser::TypeofOpContext *ctx) = 0;

  virtual void enterPostfixExpr(LangParser::PostfixExprContext *ctx) = 0;
  virtual void exitPostfixExpr(LangParser::PostfixExprContext *ctx) = 0;

  virtual void enterFunctionCall(LangParser::FunctionCallContext *ctx) = 0;
  virtual void exitFunctionCall(LangParser::FunctionCallContext *ctx) = 0;

  virtual void enterIntLiteral(LangParser::IntLiteralContext *ctx) = 0;
  virtual void exitIntLiteral(LangParser::IntLiteralContext *ctx) = 0;

  virtual void enterDoubleLiteral(LangParser::DoubleLiteralContext *ctx) = 0;
  virtual void exitDoubleLiteral(LangParser::DoubleLiteralContext *ctx) = 0;

  virtual void enterStringLiteral(LangParser::StringLiteralContext *ctx) = 0;
  virtual void exitStringLiteral(LangParser::StringLiteralContext *ctx) = 0;

  virtual void enterBoolTrue(LangParser::BoolTrueContext *ctx) = 0;
  virtual void exitBoolTrue(LangParser::BoolTrueContext *ctx) = 0;

  virtual void enterBoolFalse(LangParser::BoolFalseContext *ctx) = 0;
  virtual void exitBoolFalse(LangParser::BoolFalseContext *ctx) = 0;

  virtual void enterVariable(LangParser::VariableContext *ctx) = 0;
  virtual void exitVariable(LangParser::VariableContext *ctx) = 0;

  virtual void enterParenthesized(LangParser::ParenthesizedContext *ctx) = 0;
  virtual void exitParenthesized(LangParser::ParenthesizedContext *ctx) = 0;

  virtual void enterAnonymousFunction(LangParser::AnonymousFunctionContext *ctx) = 0;
  virtual void exitAnonymousFunction(LangParser::AnonymousFunctionContext *ctx) = 0;


};

