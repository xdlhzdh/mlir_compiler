
// Generated from Lang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "LangParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by LangParser.
 */
class  LangVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by LangParser.
   */
    virtual std::any visitProgram(LangParser::ProgramContext *context) = 0;

    virtual std::any visitVarDeclStmt(LangParser::VarDeclStmtContext *context) = 0;

    virtual std::any visitFuncDeclStmt(LangParser::FuncDeclStmtContext *context) = 0;

    virtual std::any visitIfStmt(LangParser::IfStmtContext *context) = 0;

    virtual std::any visitWhileStmt(LangParser::WhileStmtContext *context) = 0;

    virtual std::any visitPrintStmt(LangParser::PrintStmtContext *context) = 0;

    virtual std::any visitReturnStmt(LangParser::ReturnStmtContext *context) = 0;

    virtual std::any visitBreakStmt(LangParser::BreakStmtContext *context) = 0;

    virtual std::any visitContinueStmt(LangParser::ContinueStmtContext *context) = 0;

    virtual std::any visitBlockStmt(LangParser::BlockStmtContext *context) = 0;

    virtual std::any visitAssignStmt(LangParser::AssignStmtContext *context) = 0;

    virtual std::any visitExprStmt(LangParser::ExprStmtContext *context) = 0;

    virtual std::any visitBlock(LangParser::BlockContext *context) = 0;

    virtual std::any visitVarDecl(LangParser::VarDeclContext *context) = 0;

    virtual std::any visitFunctionDecl(LangParser::FunctionDeclContext *context) = 0;

    virtual std::any visitIfStatement(LangParser::IfStatementContext *context) = 0;

    virtual std::any visitWhileStatement(LangParser::WhileStatementContext *context) = 0;

    virtual std::any visitPrintStatement(LangParser::PrintStatementContext *context) = 0;

    virtual std::any visitReturnStatement(LangParser::ReturnStatementContext *context) = 0;

    virtual std::any visitAssignmentStatement(LangParser::AssignmentStatementContext *context) = 0;

    virtual std::any visitParameterList(LangParser::ParameterListContext *context) = 0;

    virtual std::any visitExpressionList(LangParser::ExpressionListContext *context) = 0;

    virtual std::any visitExpression(LangParser::ExpressionContext *context) = 0;

    virtual std::any visitTernaryExpression(LangParser::TernaryExpressionContext *context) = 0;

    virtual std::any visitLogicalOr(LangParser::LogicalOrContext *context) = 0;

    virtual std::any visitLogicalAnd(LangParser::LogicalAndContext *context) = 0;

    virtual std::any visitEquality(LangParser::EqualityContext *context) = 0;

    virtual std::any visitRelational(LangParser::RelationalContext *context) = 0;

    virtual std::any visitAdditive(LangParser::AdditiveContext *context) = 0;

    virtual std::any visitMultiplicative(LangParser::MultiplicativeContext *context) = 0;

    virtual std::any visitUnaryOp(LangParser::UnaryOpContext *context) = 0;

    virtual std::any visitTypeofOp(LangParser::TypeofOpContext *context) = 0;

    virtual std::any visitPostfixExpr(LangParser::PostfixExprContext *context) = 0;

    virtual std::any visitFunctionCall(LangParser::FunctionCallContext *context) = 0;

    virtual std::any visitIntLiteral(LangParser::IntLiteralContext *context) = 0;

    virtual std::any visitDoubleLiteral(LangParser::DoubleLiteralContext *context) = 0;

    virtual std::any visitStringLiteral(LangParser::StringLiteralContext *context) = 0;

    virtual std::any visitBoolTrue(LangParser::BoolTrueContext *context) = 0;

    virtual std::any visitBoolFalse(LangParser::BoolFalseContext *context) = 0;

    virtual std::any visitVariable(LangParser::VariableContext *context) = 0;

    virtual std::any visitParenthesized(LangParser::ParenthesizedContext *context) = 0;

    virtual std::any visitAnonymousFunction(LangParser::AnonymousFunctionContext *context) = 0;


};

