
// Generated from Lang.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  LangLexer : public antlr4::Lexer {
public:
  enum {
    LET = 1, CONST = 2, FN = 3, RETURN = 4, IF = 5, ELSE = 6, WHILE = 7, 
    BREAK = 8, CONTINUE = 9, PRINT = 10, TYPEOF = 11, TRUE = 12, FALSE = 13, 
    ID = 14, INT = 15, DOUBLE = 16, STRING = 17, PLUS = 18, MINUS = 19, 
    MUL = 20, DIV = 21, MOD = 22, DEC = 23, EQ = 24, NEQ = 25, LT = 26, 
    GT = 27, LE = 28, GE = 29, AND = 30, OR = 31, NOT = 32, ASSIGN = 33, 
    ADD_ASSIGN = 34, SUB_ASSIGN = 35, MUL_ASSIGN = 36, DIV_ASSIGN = 37, 
    ARROW = 38, QUESTION = 39, COLON = 40, LPAREN = 41, RPAREN = 42, LBRACE = 43, 
    RBRACE = 44, SEMI = 45, COMMA = 46, WS = 47, COMMENT = 48
  };

  explicit LangLexer(antlr4::CharStream *input);

  ~LangLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

