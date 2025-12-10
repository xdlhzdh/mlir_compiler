
// Generated from Lang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "LangListener.h"


/**
 * This class provides an empty implementation of LangListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  LangBaseListener : public LangListener {
public:

  virtual void enterProgram(LangParser::ProgramContext * /*ctx*/) override { }
  virtual void exitProgram(LangParser::ProgramContext * /*ctx*/) override { }

  virtual void enterVarDeclStmt(LangParser::VarDeclStmtContext * /*ctx*/) override { }
  virtual void exitVarDeclStmt(LangParser::VarDeclStmtContext * /*ctx*/) override { }

  virtual void enterFuncDeclStmt(LangParser::FuncDeclStmtContext * /*ctx*/) override { }
  virtual void exitFuncDeclStmt(LangParser::FuncDeclStmtContext * /*ctx*/) override { }

  virtual void enterIfStmt(LangParser::IfStmtContext * /*ctx*/) override { }
  virtual void exitIfStmt(LangParser::IfStmtContext * /*ctx*/) override { }

  virtual void enterWhileStmt(LangParser::WhileStmtContext * /*ctx*/) override { }
  virtual void exitWhileStmt(LangParser::WhileStmtContext * /*ctx*/) override { }

  virtual void enterPrintStmt(LangParser::PrintStmtContext * /*ctx*/) override { }
  virtual void exitPrintStmt(LangParser::PrintStmtContext * /*ctx*/) override { }

  virtual void enterReturnStmt(LangParser::ReturnStmtContext * /*ctx*/) override { }
  virtual void exitReturnStmt(LangParser::ReturnStmtContext * /*ctx*/) override { }

  virtual void enterBreakStmt(LangParser::BreakStmtContext * /*ctx*/) override { }
  virtual void exitBreakStmt(LangParser::BreakStmtContext * /*ctx*/) override { }

  virtual void enterContinueStmt(LangParser::ContinueStmtContext * /*ctx*/) override { }
  virtual void exitContinueStmt(LangParser::ContinueStmtContext * /*ctx*/) override { }

  virtual void enterBlockStmt(LangParser::BlockStmtContext * /*ctx*/) override { }
  virtual void exitBlockStmt(LangParser::BlockStmtContext * /*ctx*/) override { }

  virtual void enterAssignStmt(LangParser::AssignStmtContext * /*ctx*/) override { }
  virtual void exitAssignStmt(LangParser::AssignStmtContext * /*ctx*/) override { }

  virtual void enterExprStmt(LangParser::ExprStmtContext * /*ctx*/) override { }
  virtual void exitExprStmt(LangParser::ExprStmtContext * /*ctx*/) override { }

  virtual void enterBlock(LangParser::BlockContext * /*ctx*/) override { }
  virtual void exitBlock(LangParser::BlockContext * /*ctx*/) override { }

  virtual void enterVarDecl(LangParser::VarDeclContext * /*ctx*/) override { }
  virtual void exitVarDecl(LangParser::VarDeclContext * /*ctx*/) override { }

  virtual void enterFunctionDecl(LangParser::FunctionDeclContext * /*ctx*/) override { }
  virtual void exitFunctionDecl(LangParser::FunctionDeclContext * /*ctx*/) override { }

  virtual void enterIfStatement(LangParser::IfStatementContext * /*ctx*/) override { }
  virtual void exitIfStatement(LangParser::IfStatementContext * /*ctx*/) override { }

  virtual void enterWhileStatement(LangParser::WhileStatementContext * /*ctx*/) override { }
  virtual void exitWhileStatement(LangParser::WhileStatementContext * /*ctx*/) override { }

  virtual void enterPrintStatement(LangParser::PrintStatementContext * /*ctx*/) override { }
  virtual void exitPrintStatement(LangParser::PrintStatementContext * /*ctx*/) override { }

  virtual void enterReturnStatement(LangParser::ReturnStatementContext * /*ctx*/) override { }
  virtual void exitReturnStatement(LangParser::ReturnStatementContext * /*ctx*/) override { }

  virtual void enterAssignmentStatement(LangParser::AssignmentStatementContext * /*ctx*/) override { }
  virtual void exitAssignmentStatement(LangParser::AssignmentStatementContext * /*ctx*/) override { }

  virtual void enterParameterList(LangParser::ParameterListContext * /*ctx*/) override { }
  virtual void exitParameterList(LangParser::ParameterListContext * /*ctx*/) override { }

  virtual void enterExpressionList(LangParser::ExpressionListContext * /*ctx*/) override { }
  virtual void exitExpressionList(LangParser::ExpressionListContext * /*ctx*/) override { }

  virtual void enterExpression(LangParser::ExpressionContext * /*ctx*/) override { }
  virtual void exitExpression(LangParser::ExpressionContext * /*ctx*/) override { }

  virtual void enterTernaryExpression(LangParser::TernaryExpressionContext * /*ctx*/) override { }
  virtual void exitTernaryExpression(LangParser::TernaryExpressionContext * /*ctx*/) override { }

  virtual void enterLogicalOr(LangParser::LogicalOrContext * /*ctx*/) override { }
  virtual void exitLogicalOr(LangParser::LogicalOrContext * /*ctx*/) override { }

  virtual void enterLogicalAnd(LangParser::LogicalAndContext * /*ctx*/) override { }
  virtual void exitLogicalAnd(LangParser::LogicalAndContext * /*ctx*/) override { }

  virtual void enterEquality(LangParser::EqualityContext * /*ctx*/) override { }
  virtual void exitEquality(LangParser::EqualityContext * /*ctx*/) override { }

  virtual void enterRelational(LangParser::RelationalContext * /*ctx*/) override { }
  virtual void exitRelational(LangParser::RelationalContext * /*ctx*/) override { }

  virtual void enterAdditive(LangParser::AdditiveContext * /*ctx*/) override { }
  virtual void exitAdditive(LangParser::AdditiveContext * /*ctx*/) override { }

  virtual void enterMultiplicative(LangParser::MultiplicativeContext * /*ctx*/) override { }
  virtual void exitMultiplicative(LangParser::MultiplicativeContext * /*ctx*/) override { }

  virtual void enterUnaryOp(LangParser::UnaryOpContext * /*ctx*/) override { }
  virtual void exitUnaryOp(LangParser::UnaryOpContext * /*ctx*/) override { }

  virtual void enterTypeofOp(LangParser::TypeofOpContext * /*ctx*/) override { }
  virtual void exitTypeofOp(LangParser::TypeofOpContext * /*ctx*/) override { }

  virtual void enterPostfixExpr(LangParser::PostfixExprContext * /*ctx*/) override { }
  virtual void exitPostfixExpr(LangParser::PostfixExprContext * /*ctx*/) override { }

  virtual void enterFunctionCall(LangParser::FunctionCallContext * /*ctx*/) override { }
  virtual void exitFunctionCall(LangParser::FunctionCallContext * /*ctx*/) override { }

  virtual void enterIntLiteral(LangParser::IntLiteralContext * /*ctx*/) override { }
  virtual void exitIntLiteral(LangParser::IntLiteralContext * /*ctx*/) override { }

  virtual void enterDoubleLiteral(LangParser::DoubleLiteralContext * /*ctx*/) override { }
  virtual void exitDoubleLiteral(LangParser::DoubleLiteralContext * /*ctx*/) override { }

  virtual void enterStringLiteral(LangParser::StringLiteralContext * /*ctx*/) override { }
  virtual void exitStringLiteral(LangParser::StringLiteralContext * /*ctx*/) override { }

  virtual void enterBoolTrue(LangParser::BoolTrueContext * /*ctx*/) override { }
  virtual void exitBoolTrue(LangParser::BoolTrueContext * /*ctx*/) override { }

  virtual void enterBoolFalse(LangParser::BoolFalseContext * /*ctx*/) override { }
  virtual void exitBoolFalse(LangParser::BoolFalseContext * /*ctx*/) override { }

  virtual void enterVariable(LangParser::VariableContext * /*ctx*/) override { }
  virtual void exitVariable(LangParser::VariableContext * /*ctx*/) override { }

  virtual void enterParenthesized(LangParser::ParenthesizedContext * /*ctx*/) override { }
  virtual void exitParenthesized(LangParser::ParenthesizedContext * /*ctx*/) override { }

  virtual void enterAnonymousFunction(LangParser::AnonymousFunctionContext * /*ctx*/) override { }
  virtual void exitAnonymousFunction(LangParser::AnonymousFunctionContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

