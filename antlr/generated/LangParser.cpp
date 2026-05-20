
// Generated from Lang.g4 by ANTLR 4.13.2


#include "LangListener.h"
#include "LangVisitor.h"

#include "LangParser.h"

#include <mutex>


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct LangParserStaticData final {
  LangParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  LangParserStaticData(const LangParserStaticData&) = delete;
  LangParserStaticData(LangParserStaticData&&) = delete;
  LangParserStaticData& operator=(const LangParserStaticData&) = delete;
  LangParserStaticData& operator=(LangParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

std::once_flag langParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<LangParserStaticData> langParserStaticData = nullptr;

void langParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (langParserStaticData != nullptr) {
    return;
  }
#else
  assert(langParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<LangParserStaticData>(
    std::vector<std::string>{
      "program", "statement", "blockStatement", "varDecl", "functionDecl", 
      "ifStatement", "whileStatement", "printStatement", "returnStatement", 
      "assignmentStatement", "parameterList", "expressionList", "expression", 
      "ternaryExpression", "logicalOr", "logicalAnd", "equality", "relational", 
      "additive", "multiplicative", "unary", "postfix", "primary"
    },
    std::vector<std::string>{
      "", "'let'", "'const'", "'fn'", "'return'", "'if'", "'else'", "'while'", 
      "'break'", "'continue'", "'print'", "'typeof'", "'true'", "'false'", 
      "", "", "", "", "'+'", "'-'", "'*'", "'/'", "'%'", "'--'", "'=='", 
      "'!='", "'<'", "'>'", "'<='", "'>='", "'&&'", "'||'", "'!'", "'='", 
      "'+='", "'-='", "'*='", "'/='", "'=>'", "'\\u003F'", "':'", "'('", 
      "')'", "'{'", "'}'", "';'", "','"
    },
    std::vector<std::string>{
      "", "LET", "CONST", "FN", "RETURN", "IF", "ELSE", "WHILE", "BREAK", 
      "CONTINUE", "PRINT", "TYPEOF", "TRUE", "FALSE", "ID", "INT", "DOUBLE", 
      "STRING", "PLUS", "MINUS", "MUL", "DIV", "MOD", "DEC", "EQ", "NEQ", 
      "LT", "GT", "LE", "GE", "AND", "OR", "NOT", "ASSIGN", "ADD_ASSIGN", 
      "SUB_ASSIGN", "MUL_ASSIGN", "DIV_ASSIGN", "ARROW", "QUESTION", "COLON", 
      "LPAREN", "RPAREN", "LBRACE", "RBRACE", "SEMI", "COMMA", "WS", "COMMENT"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,48,244,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,7,
  	21,2,22,7,22,1,0,5,0,48,8,0,10,0,12,0,51,9,0,1,0,1,0,1,1,1,1,1,1,1,1,
  	1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  	1,1,1,3,1,78,8,1,1,2,1,2,5,2,82,8,2,10,2,12,2,85,9,2,1,2,1,2,1,3,1,3,
  	1,3,1,3,1,3,1,4,1,4,1,4,1,4,3,4,98,8,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,
  	5,1,5,1,5,3,5,110,8,5,1,6,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,7,1,7,1,8,
  	1,8,3,8,125,8,8,1,9,1,9,1,9,1,9,1,10,1,10,1,10,5,10,134,8,10,10,10,12,
  	10,137,9,10,1,11,1,11,1,11,5,11,142,8,11,10,11,12,11,145,9,11,1,12,1,
  	12,1,13,1,13,1,13,1,13,1,13,1,13,3,13,155,8,13,1,14,1,14,1,14,5,14,160,
  	8,14,10,14,12,14,163,9,14,1,15,1,15,1,15,5,15,168,8,15,10,15,12,15,171,
  	9,15,1,16,1,16,1,16,5,16,176,8,16,10,16,12,16,179,9,16,1,17,1,17,1,17,
  	5,17,184,8,17,10,17,12,17,187,9,17,1,18,1,18,1,18,5,18,192,8,18,10,18,
  	12,18,195,9,18,1,19,1,19,1,19,5,19,200,8,19,10,19,12,19,203,9,19,1,20,
  	1,20,1,20,1,20,1,20,3,20,210,8,20,1,21,1,21,1,21,3,21,215,8,21,1,21,1,
  	21,5,21,219,8,21,10,21,12,21,222,9,21,1,22,1,22,1,22,1,22,1,22,1,22,1,
  	22,1,22,1,22,1,22,1,22,1,22,1,22,3,22,237,8,22,1,22,1,22,1,22,3,22,242,
  	8,22,1,22,0,0,23,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,
  	38,40,42,44,0,7,1,0,1,2,1,0,33,37,1,0,24,25,1,0,26,29,1,0,18,19,1,0,20,
  	22,3,0,19,19,23,23,32,32,257,0,49,1,0,0,0,2,77,1,0,0,0,4,79,1,0,0,0,6,
  	88,1,0,0,0,8,93,1,0,0,0,10,102,1,0,0,0,12,111,1,0,0,0,14,117,1,0,0,0,
  	16,122,1,0,0,0,18,126,1,0,0,0,20,130,1,0,0,0,22,138,1,0,0,0,24,146,1,
  	0,0,0,26,148,1,0,0,0,28,156,1,0,0,0,30,164,1,0,0,0,32,172,1,0,0,0,34,
  	180,1,0,0,0,36,188,1,0,0,0,38,196,1,0,0,0,40,209,1,0,0,0,42,211,1,0,0,
  	0,44,241,1,0,0,0,46,48,3,2,1,0,47,46,1,0,0,0,48,51,1,0,0,0,49,47,1,0,
  	0,0,49,50,1,0,0,0,50,52,1,0,0,0,51,49,1,0,0,0,52,53,5,0,0,1,53,1,1,0,
  	0,0,54,55,3,6,3,0,55,56,5,45,0,0,56,78,1,0,0,0,57,78,3,8,4,0,58,78,3,
  	10,5,0,59,78,3,12,6,0,60,61,3,14,7,0,61,62,5,45,0,0,62,78,1,0,0,0,63,
  	64,3,16,8,0,64,65,5,45,0,0,65,78,1,0,0,0,66,67,5,8,0,0,67,78,5,45,0,0,
  	68,69,5,9,0,0,69,78,5,45,0,0,70,78,3,4,2,0,71,72,3,18,9,0,72,73,5,45,
  	0,0,73,78,1,0,0,0,74,75,3,24,12,0,75,76,5,45,0,0,76,78,1,0,0,0,77,54,
  	1,0,0,0,77,57,1,0,0,0,77,58,1,0,0,0,77,59,1,0,0,0,77,60,1,0,0,0,77,63,
  	1,0,0,0,77,66,1,0,0,0,77,68,1,0,0,0,77,70,1,0,0,0,77,71,1,0,0,0,77,74,
  	1,0,0,0,78,3,1,0,0,0,79,83,5,43,0,0,80,82,3,2,1,0,81,80,1,0,0,0,82,85,
  	1,0,0,0,83,81,1,0,0,0,83,84,1,0,0,0,84,86,1,0,0,0,85,83,1,0,0,0,86,87,
  	5,44,0,0,87,5,1,0,0,0,88,89,7,0,0,0,89,90,5,14,0,0,90,91,5,33,0,0,91,
  	92,3,24,12,0,92,7,1,0,0,0,93,94,5,3,0,0,94,95,5,14,0,0,95,97,5,41,0,0,
  	96,98,3,20,10,0,97,96,1,0,0,0,97,98,1,0,0,0,98,99,1,0,0,0,99,100,5,42,
  	0,0,100,101,3,4,2,0,101,9,1,0,0,0,102,103,5,5,0,0,103,104,5,41,0,0,104,
  	105,3,24,12,0,105,106,5,42,0,0,106,109,3,2,1,0,107,108,5,6,0,0,108,110,
  	3,2,1,0,109,107,1,0,0,0,109,110,1,0,0,0,110,11,1,0,0,0,111,112,5,7,0,
  	0,112,113,5,41,0,0,113,114,3,24,12,0,114,115,5,42,0,0,115,116,3,2,1,0,
  	116,13,1,0,0,0,117,118,5,10,0,0,118,119,5,41,0,0,119,120,3,22,11,0,120,
  	121,5,42,0,0,121,15,1,0,0,0,122,124,5,4,0,0,123,125,3,24,12,0,124,123,
  	1,0,0,0,124,125,1,0,0,0,125,17,1,0,0,0,126,127,5,14,0,0,127,128,7,1,0,
  	0,128,129,3,24,12,0,129,19,1,0,0,0,130,135,5,14,0,0,131,132,5,46,0,0,
  	132,134,5,14,0,0,133,131,1,0,0,0,134,137,1,0,0,0,135,133,1,0,0,0,135,
  	136,1,0,0,0,136,21,1,0,0,0,137,135,1,0,0,0,138,143,3,24,12,0,139,140,
  	5,46,0,0,140,142,3,24,12,0,141,139,1,0,0,0,142,145,1,0,0,0,143,141,1,
  	0,0,0,143,144,1,0,0,0,144,23,1,0,0,0,145,143,1,0,0,0,146,147,3,26,13,
  	0,147,25,1,0,0,0,148,154,3,28,14,0,149,150,5,39,0,0,150,151,3,24,12,0,
  	151,152,5,40,0,0,152,153,3,24,12,0,153,155,1,0,0,0,154,149,1,0,0,0,154,
  	155,1,0,0,0,155,27,1,0,0,0,156,161,3,30,15,0,157,158,5,31,0,0,158,160,
  	3,30,15,0,159,157,1,0,0,0,160,163,1,0,0,0,161,159,1,0,0,0,161,162,1,0,
  	0,0,162,29,1,0,0,0,163,161,1,0,0,0,164,169,3,32,16,0,165,166,5,30,0,0,
  	166,168,3,32,16,0,167,165,1,0,0,0,168,171,1,0,0,0,169,167,1,0,0,0,169,
  	170,1,0,0,0,170,31,1,0,0,0,171,169,1,0,0,0,172,177,3,34,17,0,173,174,
  	7,2,0,0,174,176,3,34,17,0,175,173,1,0,0,0,176,179,1,0,0,0,177,175,1,0,
  	0,0,177,178,1,0,0,0,178,33,1,0,0,0,179,177,1,0,0,0,180,185,3,36,18,0,
  	181,182,7,3,0,0,182,184,3,36,18,0,183,181,1,0,0,0,184,187,1,0,0,0,185,
  	183,1,0,0,0,185,186,1,0,0,0,186,35,1,0,0,0,187,185,1,0,0,0,188,193,3,
  	38,19,0,189,190,7,4,0,0,190,192,3,38,19,0,191,189,1,0,0,0,192,195,1,0,
  	0,0,193,191,1,0,0,0,193,194,1,0,0,0,194,37,1,0,0,0,195,193,1,0,0,0,196,
  	201,3,40,20,0,197,198,7,5,0,0,198,200,3,40,20,0,199,197,1,0,0,0,200,203,
  	1,0,0,0,201,199,1,0,0,0,201,202,1,0,0,0,202,39,1,0,0,0,203,201,1,0,0,
  	0,204,205,7,6,0,0,205,210,3,40,20,0,206,207,5,11,0,0,207,210,3,40,20,
  	0,208,210,3,42,21,0,209,204,1,0,0,0,209,206,1,0,0,0,209,208,1,0,0,0,210,
  	41,1,0,0,0,211,220,3,44,22,0,212,214,5,41,0,0,213,215,3,22,11,0,214,213,
  	1,0,0,0,214,215,1,0,0,0,215,216,1,0,0,0,216,219,5,42,0,0,217,219,5,23,
  	0,0,218,212,1,0,0,0,218,217,1,0,0,0,219,222,1,0,0,0,220,218,1,0,0,0,220,
  	221,1,0,0,0,221,43,1,0,0,0,222,220,1,0,0,0,223,242,5,15,0,0,224,242,5,
  	16,0,0,225,242,5,17,0,0,226,242,5,12,0,0,227,242,5,13,0,0,228,242,5,14,
  	0,0,229,230,5,41,0,0,230,231,3,24,12,0,231,232,5,42,0,0,232,242,1,0,0,
  	0,233,234,5,3,0,0,234,236,5,41,0,0,235,237,3,20,10,0,236,235,1,0,0,0,
  	236,237,1,0,0,0,237,238,1,0,0,0,238,239,5,42,0,0,239,240,5,38,0,0,240,
  	242,3,4,2,0,241,223,1,0,0,0,241,224,1,0,0,0,241,225,1,0,0,0,241,226,1,
  	0,0,0,241,227,1,0,0,0,241,228,1,0,0,0,241,229,1,0,0,0,241,233,1,0,0,0,
  	242,45,1,0,0,0,21,49,77,83,97,109,124,135,143,154,161,169,177,185,193,
  	201,209,214,218,220,236,241
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  langParserStaticData = std::move(staticData);
}

}

LangParser::LangParser(TokenStream *input) : LangParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

LangParser::LangParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  LangParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *langParserStaticData->atn, langParserStaticData->decisionToDFA, langParserStaticData->sharedContextCache, options);
}

LangParser::~LangParser() {
  delete _interpreter;
}

const atn::ATN& LangParser::getATN() const {
  return *langParserStaticData->atn;
}

std::string LangParser::getGrammarFileName() const {
  return "Lang.g4";
}

const std::vector<std::string>& LangParser::getRuleNames() const {
  return langParserStaticData->ruleNames;
}

const dfa::Vocabulary& LangParser::getVocabulary() const {
  return langParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView LangParser::getSerializedATN() const {
  return langParserStaticData->serializedATN;
}


//----------------- ProgramContext ------------------------------------------------------------------

LangParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::ProgramContext::EOF() {
  return getToken(LangParser::EOF, 0);
}

std::vector<LangParser::StatementContext *> LangParser::ProgramContext::statement() {
  return getRuleContexts<LangParser::StatementContext>();
}

LangParser::StatementContext* LangParser::ProgramContext::statement(size_t i) {
  return getRuleContext<LangParser::StatementContext>(i);
}


size_t LangParser::ProgramContext::getRuleIndex() const {
  return LangParser::RuleProgram;
}

void LangParser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void LangParser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


std::any LangParser::ProgramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitProgram(this);
  else
    return visitor->visitChildren(this);
}

LangParser::ProgramContext* LangParser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, LangParser::RuleProgram);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(49);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 10999420420030) != 0)) {
      setState(46);
      statement();
      setState(51);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(52);
    match(LangParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

LangParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t LangParser::StatementContext::getRuleIndex() const {
  return LangParser::RuleStatement;
}

void LangParser::StatementContext::copyFrom(StatementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- ContinueStmtContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::ContinueStmtContext::CONTINUE() {
  return getToken(LangParser::CONTINUE, 0);
}

tree::TerminalNode* LangParser::ContinueStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::ContinueStmtContext::ContinueStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::ContinueStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterContinueStmt(this);
}
void LangParser::ContinueStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitContinueStmt(this);
}

std::any LangParser::ContinueStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitContinueStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IfStmtContext ------------------------------------------------------------------

LangParser::IfStatementContext* LangParser::IfStmtContext::ifStatement() {
  return getRuleContext<LangParser::IfStatementContext>(0);
}

LangParser::IfStmtContext::IfStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::IfStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfStmt(this);
}
void LangParser::IfStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfStmt(this);
}

std::any LangParser::IfStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitIfStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- PrintStmtContext ------------------------------------------------------------------

LangParser::PrintStatementContext* LangParser::PrintStmtContext::printStatement() {
  return getRuleContext<LangParser::PrintStatementContext>(0);
}

tree::TerminalNode* LangParser::PrintStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::PrintStmtContext::PrintStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::PrintStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrintStmt(this);
}
void LangParser::PrintStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrintStmt(this);
}

std::any LangParser::PrintStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitPrintStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ExprStmtContext ------------------------------------------------------------------

LangParser::ExpressionContext* LangParser::ExprStmtContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}

tree::TerminalNode* LangParser::ExprStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::ExprStmtContext::ExprStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::ExprStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExprStmt(this);
}
void LangParser::ExprStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExprStmt(this);
}

std::any LangParser::ExprStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitExprStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- WhileStmtContext ------------------------------------------------------------------

LangParser::WhileStatementContext* LangParser::WhileStmtContext::whileStatement() {
  return getRuleContext<LangParser::WhileStatementContext>(0);
}

LangParser::WhileStmtContext::WhileStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::WhileStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhileStmt(this);
}
void LangParser::WhileStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhileStmt(this);
}

std::any LangParser::WhileStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitWhileStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- VarDeclStmtContext ------------------------------------------------------------------

LangParser::VarDeclContext* LangParser::VarDeclStmtContext::varDecl() {
  return getRuleContext<LangParser::VarDeclContext>(0);
}

tree::TerminalNode* LangParser::VarDeclStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::VarDeclStmtContext::VarDeclStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::VarDeclStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVarDeclStmt(this);
}
void LangParser::VarDeclStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVarDeclStmt(this);
}

std::any LangParser::VarDeclStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitVarDeclStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BreakStmtContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::BreakStmtContext::BREAK() {
  return getToken(LangParser::BREAK, 0);
}

tree::TerminalNode* LangParser::BreakStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::BreakStmtContext::BreakStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::BreakStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBreakStmt(this);
}
void LangParser::BreakStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBreakStmt(this);
}

std::any LangParser::BreakStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitBreakStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AssignStmtContext ------------------------------------------------------------------

LangParser::AssignmentStatementContext* LangParser::AssignStmtContext::assignmentStatement() {
  return getRuleContext<LangParser::AssignmentStatementContext>(0);
}

tree::TerminalNode* LangParser::AssignStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::AssignStmtContext::AssignStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::AssignStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignStmt(this);
}
void LangParser::AssignStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignStmt(this);
}

std::any LangParser::AssignStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitAssignStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BlockStmtContext ------------------------------------------------------------------

LangParser::BlockStatementContext* LangParser::BlockStmtContext::blockStatement() {
  return getRuleContext<LangParser::BlockStatementContext>(0);
}

LangParser::BlockStmtContext::BlockStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::BlockStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBlockStmt(this);
}
void LangParser::BlockStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBlockStmt(this);
}

std::any LangParser::BlockStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitBlockStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- FuncDeclStmtContext ------------------------------------------------------------------

LangParser::FunctionDeclContext* LangParser::FuncDeclStmtContext::functionDecl() {
  return getRuleContext<LangParser::FunctionDeclContext>(0);
}

LangParser::FuncDeclStmtContext::FuncDeclStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::FuncDeclStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFuncDeclStmt(this);
}
void LangParser::FuncDeclStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFuncDeclStmt(this);
}

std::any LangParser::FuncDeclStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitFuncDeclStmt(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ReturnStmtContext ------------------------------------------------------------------

LangParser::ReturnStatementContext* LangParser::ReturnStmtContext::returnStatement() {
  return getRuleContext<LangParser::ReturnStatementContext>(0);
}

tree::TerminalNode* LangParser::ReturnStmtContext::SEMI() {
  return getToken(LangParser::SEMI, 0);
}

LangParser::ReturnStmtContext::ReturnStmtContext(StatementContext *ctx) { copyFrom(ctx); }

void LangParser::ReturnStmtContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturnStmt(this);
}
void LangParser::ReturnStmtContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturnStmt(this);
}

std::any LangParser::ReturnStmtContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitReturnStmt(this);
  else
    return visitor->visitChildren(this);
}
LangParser::StatementContext* LangParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 2, LangParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(77);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<LangParser::VarDeclStmtContext>(_localctx);
      enterOuterAlt(_localctx, 1);
      setState(54);
      varDecl();
      setState(55);
      match(LangParser::SEMI);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<LangParser::FuncDeclStmtContext>(_localctx);
      enterOuterAlt(_localctx, 2);
      setState(57);
      functionDecl();
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<LangParser::IfStmtContext>(_localctx);
      enterOuterAlt(_localctx, 3);
      setState(58);
      ifStatement();
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<LangParser::WhileStmtContext>(_localctx);
      enterOuterAlt(_localctx, 4);
      setState(59);
      whileStatement();
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<LangParser::PrintStmtContext>(_localctx);
      enterOuterAlt(_localctx, 5);
      setState(60);
      printStatement();
      setState(61);
      match(LangParser::SEMI);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<LangParser::ReturnStmtContext>(_localctx);
      enterOuterAlt(_localctx, 6);
      setState(63);
      returnStatement();
      setState(64);
      match(LangParser::SEMI);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<LangParser::BreakStmtContext>(_localctx);
      enterOuterAlt(_localctx, 7);
      setState(66);
      match(LangParser::BREAK);
      setState(67);
      match(LangParser::SEMI);
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<LangParser::ContinueStmtContext>(_localctx);
      enterOuterAlt(_localctx, 8);
      setState(68);
      match(LangParser::CONTINUE);
      setState(69);
      match(LangParser::SEMI);
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<LangParser::BlockStmtContext>(_localctx);
      enterOuterAlt(_localctx, 9);
      setState(70);
      blockStatement();
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<LangParser::AssignStmtContext>(_localctx);
      enterOuterAlt(_localctx, 10);
      setState(71);
      assignmentStatement();
      setState(72);
      match(LangParser::SEMI);
      break;
    }

    case 11: {
      _localctx = _tracker.createInstance<LangParser::ExprStmtContext>(_localctx);
      enterOuterAlt(_localctx, 11);
      setState(74);
      expression();
      setState(75);
      match(LangParser::SEMI);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BlockStatementContext ------------------------------------------------------------------

LangParser::BlockStatementContext::BlockStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t LangParser::BlockStatementContext::getRuleIndex() const {
  return LangParser::RuleBlockStatement;
}

void LangParser::BlockStatementContext::copyFrom(BlockStatementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- BlockContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::BlockContext::LBRACE() {
  return getToken(LangParser::LBRACE, 0);
}

tree::TerminalNode* LangParser::BlockContext::RBRACE() {
  return getToken(LangParser::RBRACE, 0);
}

std::vector<LangParser::StatementContext *> LangParser::BlockContext::statement() {
  return getRuleContexts<LangParser::StatementContext>();
}

LangParser::StatementContext* LangParser::BlockContext::statement(size_t i) {
  return getRuleContext<LangParser::StatementContext>(i);
}

LangParser::BlockContext::BlockContext(BlockStatementContext *ctx) { copyFrom(ctx); }

void LangParser::BlockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBlock(this);
}
void LangParser::BlockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBlock(this);
}

std::any LangParser::BlockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitBlock(this);
  else
    return visitor->visitChildren(this);
}
LangParser::BlockStatementContext* LangParser::blockStatement() {
  BlockStatementContext *_localctx = _tracker.createInstance<BlockStatementContext>(_ctx, getState());
  enterRule(_localctx, 4, LangParser::RuleBlockStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    _localctx = _tracker.createInstance<LangParser::BlockContext>(_localctx);
    enterOuterAlt(_localctx, 1);
    setState(79);
    match(LangParser::LBRACE);
    setState(83);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 10999420420030) != 0)) {
      setState(80);
      statement();
      setState(85);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(86);
    match(LangParser::RBRACE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDeclContext ------------------------------------------------------------------

LangParser::VarDeclContext::VarDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::VarDeclContext::ID() {
  return getToken(LangParser::ID, 0);
}

tree::TerminalNode* LangParser::VarDeclContext::ASSIGN() {
  return getToken(LangParser::ASSIGN, 0);
}

LangParser::ExpressionContext* LangParser::VarDeclContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}

tree::TerminalNode* LangParser::VarDeclContext::LET() {
  return getToken(LangParser::LET, 0);
}

tree::TerminalNode* LangParser::VarDeclContext::CONST() {
  return getToken(LangParser::CONST, 0);
}


size_t LangParser::VarDeclContext::getRuleIndex() const {
  return LangParser::RuleVarDecl;
}

void LangParser::VarDeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVarDecl(this);
}

void LangParser::VarDeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVarDecl(this);
}


std::any LangParser::VarDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitVarDecl(this);
  else
    return visitor->visitChildren(this);
}

LangParser::VarDeclContext* LangParser::varDecl() {
  VarDeclContext *_localctx = _tracker.createInstance<VarDeclContext>(_ctx, getState());
  enterRule(_localctx, 6, LangParser::RuleVarDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(88);
    _la = _input->LA(1);
    if (!(_la == LangParser::LET

    || _la == LangParser::CONST)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(89);
    match(LangParser::ID);
    setState(90);
    match(LangParser::ASSIGN);
    setState(91);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FunctionDeclContext ------------------------------------------------------------------

LangParser::FunctionDeclContext::FunctionDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::FunctionDeclContext::FN() {
  return getToken(LangParser::FN, 0);
}

tree::TerminalNode* LangParser::FunctionDeclContext::ID() {
  return getToken(LangParser::ID, 0);
}

tree::TerminalNode* LangParser::FunctionDeclContext::LPAREN() {
  return getToken(LangParser::LPAREN, 0);
}

tree::TerminalNode* LangParser::FunctionDeclContext::RPAREN() {
  return getToken(LangParser::RPAREN, 0);
}

LangParser::BlockStatementContext* LangParser::FunctionDeclContext::blockStatement() {
  return getRuleContext<LangParser::BlockStatementContext>(0);
}

LangParser::ParameterListContext* LangParser::FunctionDeclContext::parameterList() {
  return getRuleContext<LangParser::ParameterListContext>(0);
}


size_t LangParser::FunctionDeclContext::getRuleIndex() const {
  return LangParser::RuleFunctionDecl;
}

void LangParser::FunctionDeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionDecl(this);
}

void LangParser::FunctionDeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionDecl(this);
}


std::any LangParser::FunctionDeclContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitFunctionDecl(this);
  else
    return visitor->visitChildren(this);
}

LangParser::FunctionDeclContext* LangParser::functionDecl() {
  FunctionDeclContext *_localctx = _tracker.createInstance<FunctionDeclContext>(_ctx, getState());
  enterRule(_localctx, 8, LangParser::RuleFunctionDecl);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(93);
    match(LangParser::FN);
    setState(94);
    match(LangParser::ID);
    setState(95);
    match(LangParser::LPAREN);
    setState(97);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == LangParser::ID) {
      setState(96);
      parameterList();
    }
    setState(99);
    match(LangParser::RPAREN);
    setState(100);
    blockStatement();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfStatementContext ------------------------------------------------------------------

LangParser::IfStatementContext::IfStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::IfStatementContext::IF() {
  return getToken(LangParser::IF, 0);
}

tree::TerminalNode* LangParser::IfStatementContext::LPAREN() {
  return getToken(LangParser::LPAREN, 0);
}

LangParser::ExpressionContext* LangParser::IfStatementContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}

tree::TerminalNode* LangParser::IfStatementContext::RPAREN() {
  return getToken(LangParser::RPAREN, 0);
}

std::vector<LangParser::StatementContext *> LangParser::IfStatementContext::statement() {
  return getRuleContexts<LangParser::StatementContext>();
}

LangParser::StatementContext* LangParser::IfStatementContext::statement(size_t i) {
  return getRuleContext<LangParser::StatementContext>(i);
}

tree::TerminalNode* LangParser::IfStatementContext::ELSE() {
  return getToken(LangParser::ELSE, 0);
}


size_t LangParser::IfStatementContext::getRuleIndex() const {
  return LangParser::RuleIfStatement;
}

void LangParser::IfStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfStatement(this);
}

void LangParser::IfStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfStatement(this);
}


std::any LangParser::IfStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitIfStatement(this);
  else
    return visitor->visitChildren(this);
}

LangParser::IfStatementContext* LangParser::ifStatement() {
  IfStatementContext *_localctx = _tracker.createInstance<IfStatementContext>(_ctx, getState());
  enterRule(_localctx, 10, LangParser::RuleIfStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(102);
    match(LangParser::IF);
    setState(103);
    match(LangParser::LPAREN);
    setState(104);
    expression();
    setState(105);
    match(LangParser::RPAREN);
    setState(106);
    statement();
    setState(109);
    _errHandler->sync(this);

    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
    case 1: {
      setState(107);
      match(LangParser::ELSE);
      setState(108);
      statement();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- WhileStatementContext ------------------------------------------------------------------

LangParser::WhileStatementContext::WhileStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::WhileStatementContext::WHILE() {
  return getToken(LangParser::WHILE, 0);
}

tree::TerminalNode* LangParser::WhileStatementContext::LPAREN() {
  return getToken(LangParser::LPAREN, 0);
}

LangParser::ExpressionContext* LangParser::WhileStatementContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}

tree::TerminalNode* LangParser::WhileStatementContext::RPAREN() {
  return getToken(LangParser::RPAREN, 0);
}

LangParser::StatementContext* LangParser::WhileStatementContext::statement() {
  return getRuleContext<LangParser::StatementContext>(0);
}


size_t LangParser::WhileStatementContext::getRuleIndex() const {
  return LangParser::RuleWhileStatement;
}

void LangParser::WhileStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterWhileStatement(this);
}

void LangParser::WhileStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitWhileStatement(this);
}


std::any LangParser::WhileStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitWhileStatement(this);
  else
    return visitor->visitChildren(this);
}

LangParser::WhileStatementContext* LangParser::whileStatement() {
  WhileStatementContext *_localctx = _tracker.createInstance<WhileStatementContext>(_ctx, getState());
  enterRule(_localctx, 12, LangParser::RuleWhileStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(111);
    match(LangParser::WHILE);
    setState(112);
    match(LangParser::LPAREN);
    setState(113);
    expression();
    setState(114);
    match(LangParser::RPAREN);
    setState(115);
    statement();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrintStatementContext ------------------------------------------------------------------

LangParser::PrintStatementContext::PrintStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::PrintStatementContext::PRINT() {
  return getToken(LangParser::PRINT, 0);
}

tree::TerminalNode* LangParser::PrintStatementContext::LPAREN() {
  return getToken(LangParser::LPAREN, 0);
}

LangParser::ExpressionListContext* LangParser::PrintStatementContext::expressionList() {
  return getRuleContext<LangParser::ExpressionListContext>(0);
}

tree::TerminalNode* LangParser::PrintStatementContext::RPAREN() {
  return getToken(LangParser::RPAREN, 0);
}


size_t LangParser::PrintStatementContext::getRuleIndex() const {
  return LangParser::RulePrintStatement;
}

void LangParser::PrintStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPrintStatement(this);
}

void LangParser::PrintStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPrintStatement(this);
}


std::any LangParser::PrintStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitPrintStatement(this);
  else
    return visitor->visitChildren(this);
}

LangParser::PrintStatementContext* LangParser::printStatement() {
  PrintStatementContext *_localctx = _tracker.createInstance<PrintStatementContext>(_ctx, getState());
  enterRule(_localctx, 14, LangParser::RulePrintStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(117);
    match(LangParser::PRINT);
    setState(118);
    match(LangParser::LPAREN);
    setState(119);
    expressionList();
    setState(120);
    match(LangParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ReturnStatementContext ------------------------------------------------------------------

LangParser::ReturnStatementContext::ReturnStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::ReturnStatementContext::RETURN() {
  return getToken(LangParser::RETURN, 0);
}

LangParser::ExpressionContext* LangParser::ReturnStatementContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}


size_t LangParser::ReturnStatementContext::getRuleIndex() const {
  return LangParser::RuleReturnStatement;
}

void LangParser::ReturnStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReturnStatement(this);
}

void LangParser::ReturnStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReturnStatement(this);
}


std::any LangParser::ReturnStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitReturnStatement(this);
  else
    return visitor->visitChildren(this);
}

LangParser::ReturnStatementContext* LangParser::returnStatement() {
  ReturnStatementContext *_localctx = _tracker.createInstance<ReturnStatementContext>(_ctx, getState());
  enterRule(_localctx, 16, LangParser::RuleReturnStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(122);
    match(LangParser::RETURN);
    setState(124);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 2203327395848) != 0)) {
      setState(123);
      expression();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AssignmentStatementContext ------------------------------------------------------------------

LangParser::AssignmentStatementContext::AssignmentStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* LangParser::AssignmentStatementContext::ID() {
  return getToken(LangParser::ID, 0);
}

LangParser::ExpressionContext* LangParser::AssignmentStatementContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}

tree::TerminalNode* LangParser::AssignmentStatementContext::ASSIGN() {
  return getToken(LangParser::ASSIGN, 0);
}

tree::TerminalNode* LangParser::AssignmentStatementContext::ADD_ASSIGN() {
  return getToken(LangParser::ADD_ASSIGN, 0);
}

tree::TerminalNode* LangParser::AssignmentStatementContext::SUB_ASSIGN() {
  return getToken(LangParser::SUB_ASSIGN, 0);
}

tree::TerminalNode* LangParser::AssignmentStatementContext::MUL_ASSIGN() {
  return getToken(LangParser::MUL_ASSIGN, 0);
}

tree::TerminalNode* LangParser::AssignmentStatementContext::DIV_ASSIGN() {
  return getToken(LangParser::DIV_ASSIGN, 0);
}


size_t LangParser::AssignmentStatementContext::getRuleIndex() const {
  return LangParser::RuleAssignmentStatement;
}

void LangParser::AssignmentStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignmentStatement(this);
}

void LangParser::AssignmentStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignmentStatement(this);
}


std::any LangParser::AssignmentStatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitAssignmentStatement(this);
  else
    return visitor->visitChildren(this);
}

LangParser::AssignmentStatementContext* LangParser::assignmentStatement() {
  AssignmentStatementContext *_localctx = _tracker.createInstance<AssignmentStatementContext>(_ctx, getState());
  enterRule(_localctx, 18, LangParser::RuleAssignmentStatement);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(126);
    match(LangParser::ID);
    setState(127);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 266287972352) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(128);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ParameterListContext ------------------------------------------------------------------

LangParser::ParameterListContext::ParameterListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> LangParser::ParameterListContext::ID() {
  return getTokens(LangParser::ID);
}

tree::TerminalNode* LangParser::ParameterListContext::ID(size_t i) {
  return getToken(LangParser::ID, i);
}

std::vector<tree::TerminalNode *> LangParser::ParameterListContext::COMMA() {
  return getTokens(LangParser::COMMA);
}

tree::TerminalNode* LangParser::ParameterListContext::COMMA(size_t i) {
  return getToken(LangParser::COMMA, i);
}


size_t LangParser::ParameterListContext::getRuleIndex() const {
  return LangParser::RuleParameterList;
}

void LangParser::ParameterListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParameterList(this);
}

void LangParser::ParameterListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParameterList(this);
}


std::any LangParser::ParameterListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitParameterList(this);
  else
    return visitor->visitChildren(this);
}

LangParser::ParameterListContext* LangParser::parameterList() {
  ParameterListContext *_localctx = _tracker.createInstance<ParameterListContext>(_ctx, getState());
  enterRule(_localctx, 20, LangParser::RuleParameterList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(130);
    match(LangParser::ID);
    setState(135);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::COMMA) {
      setState(131);
      match(LangParser::COMMA);
      setState(132);
      match(LangParser::ID);
      setState(137);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionListContext ------------------------------------------------------------------

LangParser::ExpressionListContext::ExpressionListContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::ExpressionContext *> LangParser::ExpressionListContext::expression() {
  return getRuleContexts<LangParser::ExpressionContext>();
}

LangParser::ExpressionContext* LangParser::ExpressionListContext::expression(size_t i) {
  return getRuleContext<LangParser::ExpressionContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::ExpressionListContext::COMMA() {
  return getTokens(LangParser::COMMA);
}

tree::TerminalNode* LangParser::ExpressionListContext::COMMA(size_t i) {
  return getToken(LangParser::COMMA, i);
}


size_t LangParser::ExpressionListContext::getRuleIndex() const {
  return LangParser::RuleExpressionList;
}

void LangParser::ExpressionListContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpressionList(this);
}

void LangParser::ExpressionListContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpressionList(this);
}


std::any LangParser::ExpressionListContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitExpressionList(this);
  else
    return visitor->visitChildren(this);
}

LangParser::ExpressionListContext* LangParser::expressionList() {
  ExpressionListContext *_localctx = _tracker.createInstance<ExpressionListContext>(_ctx, getState());
  enterRule(_localctx, 22, LangParser::RuleExpressionList);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(138);
    expression();
    setState(143);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::COMMA) {
      setState(139);
      match(LangParser::COMMA);
      setState(140);
      expression();
      setState(145);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

LangParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

LangParser::TernaryExpressionContext* LangParser::ExpressionContext::ternaryExpression() {
  return getRuleContext<LangParser::TernaryExpressionContext>(0);
}


size_t LangParser::ExpressionContext::getRuleIndex() const {
  return LangParser::RuleExpression;
}

void LangParser::ExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void LangParser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


std::any LangParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

LangParser::ExpressionContext* LangParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 24, LangParser::RuleExpression);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(146);
    ternaryExpression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- TernaryExpressionContext ------------------------------------------------------------------

LangParser::TernaryExpressionContext::TernaryExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

LangParser::LogicalOrContext* LangParser::TernaryExpressionContext::logicalOr() {
  return getRuleContext<LangParser::LogicalOrContext>(0);
}

tree::TerminalNode* LangParser::TernaryExpressionContext::QUESTION() {
  return getToken(LangParser::QUESTION, 0);
}

std::vector<LangParser::ExpressionContext *> LangParser::TernaryExpressionContext::expression() {
  return getRuleContexts<LangParser::ExpressionContext>();
}

LangParser::ExpressionContext* LangParser::TernaryExpressionContext::expression(size_t i) {
  return getRuleContext<LangParser::ExpressionContext>(i);
}

tree::TerminalNode* LangParser::TernaryExpressionContext::COLON() {
  return getToken(LangParser::COLON, 0);
}


size_t LangParser::TernaryExpressionContext::getRuleIndex() const {
  return LangParser::RuleTernaryExpression;
}

void LangParser::TernaryExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTernaryExpression(this);
}

void LangParser::TernaryExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTernaryExpression(this);
}


std::any LangParser::TernaryExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitTernaryExpression(this);
  else
    return visitor->visitChildren(this);
}

LangParser::TernaryExpressionContext* LangParser::ternaryExpression() {
  TernaryExpressionContext *_localctx = _tracker.createInstance<TernaryExpressionContext>(_ctx, getState());
  enterRule(_localctx, 26, LangParser::RuleTernaryExpression);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(148);
    logicalOr();
    setState(154);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == LangParser::QUESTION) {
      setState(149);
      match(LangParser::QUESTION);
      setState(150);
      expression();
      setState(151);
      match(LangParser::COLON);
      setState(152);
      expression();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalOrContext ------------------------------------------------------------------

LangParser::LogicalOrContext::LogicalOrContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::LogicalAndContext *> LangParser::LogicalOrContext::logicalAnd() {
  return getRuleContexts<LangParser::LogicalAndContext>();
}

LangParser::LogicalAndContext* LangParser::LogicalOrContext::logicalAnd(size_t i) {
  return getRuleContext<LangParser::LogicalAndContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::LogicalOrContext::OR() {
  return getTokens(LangParser::OR);
}

tree::TerminalNode* LangParser::LogicalOrContext::OR(size_t i) {
  return getToken(LangParser::OR, i);
}


size_t LangParser::LogicalOrContext::getRuleIndex() const {
  return LangParser::RuleLogicalOr;
}

void LangParser::LogicalOrContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogicalOr(this);
}

void LangParser::LogicalOrContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogicalOr(this);
}


std::any LangParser::LogicalOrContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitLogicalOr(this);
  else
    return visitor->visitChildren(this);
}

LangParser::LogicalOrContext* LangParser::logicalOr() {
  LogicalOrContext *_localctx = _tracker.createInstance<LogicalOrContext>(_ctx, getState());
  enterRule(_localctx, 28, LangParser::RuleLogicalOr);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(156);
    logicalAnd();
    setState(161);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::OR) {
      setState(157);
      match(LangParser::OR);
      setState(158);
      logicalAnd();
      setState(163);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- LogicalAndContext ------------------------------------------------------------------

LangParser::LogicalAndContext::LogicalAndContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::EqualityContext *> LangParser::LogicalAndContext::equality() {
  return getRuleContexts<LangParser::EqualityContext>();
}

LangParser::EqualityContext* LangParser::LogicalAndContext::equality(size_t i) {
  return getRuleContext<LangParser::EqualityContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::LogicalAndContext::AND() {
  return getTokens(LangParser::AND);
}

tree::TerminalNode* LangParser::LogicalAndContext::AND(size_t i) {
  return getToken(LangParser::AND, i);
}


size_t LangParser::LogicalAndContext::getRuleIndex() const {
  return LangParser::RuleLogicalAnd;
}

void LangParser::LogicalAndContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogicalAnd(this);
}

void LangParser::LogicalAndContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogicalAnd(this);
}


std::any LangParser::LogicalAndContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitLogicalAnd(this);
  else
    return visitor->visitChildren(this);
}

LangParser::LogicalAndContext* LangParser::logicalAnd() {
  LogicalAndContext *_localctx = _tracker.createInstance<LogicalAndContext>(_ctx, getState());
  enterRule(_localctx, 30, LangParser::RuleLogicalAnd);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(164);
    equality();
    setState(169);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::AND) {
      setState(165);
      match(LangParser::AND);
      setState(166);
      equality();
      setState(171);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- EqualityContext ------------------------------------------------------------------

LangParser::EqualityContext::EqualityContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::RelationalContext *> LangParser::EqualityContext::relational() {
  return getRuleContexts<LangParser::RelationalContext>();
}

LangParser::RelationalContext* LangParser::EqualityContext::relational(size_t i) {
  return getRuleContext<LangParser::RelationalContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::EqualityContext::EQ() {
  return getTokens(LangParser::EQ);
}

tree::TerminalNode* LangParser::EqualityContext::EQ(size_t i) {
  return getToken(LangParser::EQ, i);
}

std::vector<tree::TerminalNode *> LangParser::EqualityContext::NEQ() {
  return getTokens(LangParser::NEQ);
}

tree::TerminalNode* LangParser::EqualityContext::NEQ(size_t i) {
  return getToken(LangParser::NEQ, i);
}


size_t LangParser::EqualityContext::getRuleIndex() const {
  return LangParser::RuleEquality;
}

void LangParser::EqualityContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEquality(this);
}

void LangParser::EqualityContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEquality(this);
}


std::any LangParser::EqualityContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitEquality(this);
  else
    return visitor->visitChildren(this);
}

LangParser::EqualityContext* LangParser::equality() {
  EqualityContext *_localctx = _tracker.createInstance<EqualityContext>(_ctx, getState());
  enterRule(_localctx, 32, LangParser::RuleEquality);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(172);
    relational();
    setState(177);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::EQ

    || _la == LangParser::NEQ) {
      setState(173);
      _la = _input->LA(1);
      if (!(_la == LangParser::EQ

      || _la == LangParser::NEQ)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(174);
      relational();
      setState(179);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RelationalContext ------------------------------------------------------------------

LangParser::RelationalContext::RelationalContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::AdditiveContext *> LangParser::RelationalContext::additive() {
  return getRuleContexts<LangParser::AdditiveContext>();
}

LangParser::AdditiveContext* LangParser::RelationalContext::additive(size_t i) {
  return getRuleContext<LangParser::AdditiveContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::RelationalContext::LT() {
  return getTokens(LangParser::LT);
}

tree::TerminalNode* LangParser::RelationalContext::LT(size_t i) {
  return getToken(LangParser::LT, i);
}

std::vector<tree::TerminalNode *> LangParser::RelationalContext::GT() {
  return getTokens(LangParser::GT);
}

tree::TerminalNode* LangParser::RelationalContext::GT(size_t i) {
  return getToken(LangParser::GT, i);
}

std::vector<tree::TerminalNode *> LangParser::RelationalContext::LE() {
  return getTokens(LangParser::LE);
}

tree::TerminalNode* LangParser::RelationalContext::LE(size_t i) {
  return getToken(LangParser::LE, i);
}

std::vector<tree::TerminalNode *> LangParser::RelationalContext::GE() {
  return getTokens(LangParser::GE);
}

tree::TerminalNode* LangParser::RelationalContext::GE(size_t i) {
  return getToken(LangParser::GE, i);
}


size_t LangParser::RelationalContext::getRuleIndex() const {
  return LangParser::RuleRelational;
}

void LangParser::RelationalContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRelational(this);
}

void LangParser::RelationalContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRelational(this);
}


std::any LangParser::RelationalContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitRelational(this);
  else
    return visitor->visitChildren(this);
}

LangParser::RelationalContext* LangParser::relational() {
  RelationalContext *_localctx = _tracker.createInstance<RelationalContext>(_ctx, getState());
  enterRule(_localctx, 34, LangParser::RuleRelational);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(180);
    additive();
    setState(185);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 1006632960) != 0)) {
      setState(181);
      _la = _input->LA(1);
      if (!((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 1006632960) != 0))) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(182);
      additive();
      setState(187);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AdditiveContext ------------------------------------------------------------------

LangParser::AdditiveContext::AdditiveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::MultiplicativeContext *> LangParser::AdditiveContext::multiplicative() {
  return getRuleContexts<LangParser::MultiplicativeContext>();
}

LangParser::MultiplicativeContext* LangParser::AdditiveContext::multiplicative(size_t i) {
  return getRuleContext<LangParser::MultiplicativeContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::AdditiveContext::PLUS() {
  return getTokens(LangParser::PLUS);
}

tree::TerminalNode* LangParser::AdditiveContext::PLUS(size_t i) {
  return getToken(LangParser::PLUS, i);
}

std::vector<tree::TerminalNode *> LangParser::AdditiveContext::MINUS() {
  return getTokens(LangParser::MINUS);
}

tree::TerminalNode* LangParser::AdditiveContext::MINUS(size_t i) {
  return getToken(LangParser::MINUS, i);
}


size_t LangParser::AdditiveContext::getRuleIndex() const {
  return LangParser::RuleAdditive;
}

void LangParser::AdditiveContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAdditive(this);
}

void LangParser::AdditiveContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAdditive(this);
}


std::any LangParser::AdditiveContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitAdditive(this);
  else
    return visitor->visitChildren(this);
}

LangParser::AdditiveContext* LangParser::additive() {
  AdditiveContext *_localctx = _tracker.createInstance<AdditiveContext>(_ctx, getState());
  enterRule(_localctx, 36, LangParser::RuleAdditive);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(188);
    multiplicative();
    setState(193);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::PLUS

    || _la == LangParser::MINUS) {
      setState(189);
      _la = _input->LA(1);
      if (!(_la == LangParser::PLUS

      || _la == LangParser::MINUS)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(190);
      multiplicative();
      setState(195);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MultiplicativeContext ------------------------------------------------------------------

LangParser::MultiplicativeContext::MultiplicativeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<LangParser::UnaryContext *> LangParser::MultiplicativeContext::unary() {
  return getRuleContexts<LangParser::UnaryContext>();
}

LangParser::UnaryContext* LangParser::MultiplicativeContext::unary(size_t i) {
  return getRuleContext<LangParser::UnaryContext>(i);
}

std::vector<tree::TerminalNode *> LangParser::MultiplicativeContext::MUL() {
  return getTokens(LangParser::MUL);
}

tree::TerminalNode* LangParser::MultiplicativeContext::MUL(size_t i) {
  return getToken(LangParser::MUL, i);
}

std::vector<tree::TerminalNode *> LangParser::MultiplicativeContext::DIV() {
  return getTokens(LangParser::DIV);
}

tree::TerminalNode* LangParser::MultiplicativeContext::DIV(size_t i) {
  return getToken(LangParser::DIV, i);
}

std::vector<tree::TerminalNode *> LangParser::MultiplicativeContext::MOD() {
  return getTokens(LangParser::MOD);
}

tree::TerminalNode* LangParser::MultiplicativeContext::MOD(size_t i) {
  return getToken(LangParser::MOD, i);
}


size_t LangParser::MultiplicativeContext::getRuleIndex() const {
  return LangParser::RuleMultiplicative;
}

void LangParser::MultiplicativeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiplicative(this);
}

void LangParser::MultiplicativeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiplicative(this);
}


std::any LangParser::MultiplicativeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitMultiplicative(this);
  else
    return visitor->visitChildren(this);
}

LangParser::MultiplicativeContext* LangParser::multiplicative() {
  MultiplicativeContext *_localctx = _tracker.createInstance<MultiplicativeContext>(_ctx, getState());
  enterRule(_localctx, 38, LangParser::RuleMultiplicative);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(196);
    unary();
    setState(201);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 7340032) != 0)) {
      setState(197);
      _la = _input->LA(1);
      if (!((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 7340032) != 0))) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(198);
      unary();
      setState(203);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryContext ------------------------------------------------------------------

LangParser::UnaryContext::UnaryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t LangParser::UnaryContext::getRuleIndex() const {
  return LangParser::RuleUnary;
}

void LangParser::UnaryContext::copyFrom(UnaryContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- PostfixExprContext ------------------------------------------------------------------

LangParser::PostfixContext* LangParser::PostfixExprContext::postfix() {
  return getRuleContext<LangParser::PostfixContext>(0);
}

LangParser::PostfixExprContext::PostfixExprContext(UnaryContext *ctx) { copyFrom(ctx); }

void LangParser::PostfixExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPostfixExpr(this);
}
void LangParser::PostfixExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPostfixExpr(this);
}

std::any LangParser::PostfixExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitPostfixExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- UnaryOpContext ------------------------------------------------------------------

LangParser::UnaryContext* LangParser::UnaryOpContext::unary() {
  return getRuleContext<LangParser::UnaryContext>(0);
}

tree::TerminalNode* LangParser::UnaryOpContext::MINUS() {
  return getToken(LangParser::MINUS, 0);
}

tree::TerminalNode* LangParser::UnaryOpContext::NOT() {
  return getToken(LangParser::NOT, 0);
}

tree::TerminalNode* LangParser::UnaryOpContext::DEC() {
  return getToken(LangParser::DEC, 0);
}

LangParser::UnaryOpContext::UnaryOpContext(UnaryContext *ctx) { copyFrom(ctx); }

void LangParser::UnaryOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryOp(this);
}
void LangParser::UnaryOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryOp(this);
}

std::any LangParser::UnaryOpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitUnaryOp(this);
  else
    return visitor->visitChildren(this);
}
//----------------- TypeofOpContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::TypeofOpContext::TYPEOF() {
  return getToken(LangParser::TYPEOF, 0);
}

LangParser::UnaryContext* LangParser::TypeofOpContext::unary() {
  return getRuleContext<LangParser::UnaryContext>(0);
}

LangParser::TypeofOpContext::TypeofOpContext(UnaryContext *ctx) { copyFrom(ctx); }

void LangParser::TypeofOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTypeofOp(this);
}
void LangParser::TypeofOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTypeofOp(this);
}

std::any LangParser::TypeofOpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitTypeofOp(this);
  else
    return visitor->visitChildren(this);
}
LangParser::UnaryContext* LangParser::unary() {
  UnaryContext *_localctx = _tracker.createInstance<UnaryContext>(_ctx, getState());
  enterRule(_localctx, 40, LangParser::RuleUnary);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(209);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case LangParser::MINUS:
      case LangParser::DEC:
      case LangParser::NOT: {
        _localctx = _tracker.createInstance<LangParser::UnaryOpContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(204);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & 4303880192) != 0))) {
        _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(205);
        unary();
        break;
      }

      case LangParser::TYPEOF: {
        _localctx = _tracker.createInstance<LangParser::TypeofOpContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(206);
        match(LangParser::TYPEOF);
        setState(207);
        unary();
        break;
      }

      case LangParser::FN:
      case LangParser::TRUE:
      case LangParser::FALSE:
      case LangParser::ID:
      case LangParser::INT:
      case LangParser::DOUBLE:
      case LangParser::STRING:
      case LangParser::LPAREN: {
        _localctx = _tracker.createInstance<LangParser::PostfixExprContext>(_localctx);
        enterOuterAlt(_localctx, 3);
        setState(208);
        postfix();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PostfixContext ------------------------------------------------------------------

LangParser::PostfixContext::PostfixContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t LangParser::PostfixContext::getRuleIndex() const {
  return LangParser::RulePostfix;
}

void LangParser::PostfixContext::copyFrom(PostfixContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- FunctionCallContext ------------------------------------------------------------------

LangParser::PrimaryContext* LangParser::FunctionCallContext::primary() {
  return getRuleContext<LangParser::PrimaryContext>(0);
}

std::vector<tree::TerminalNode *> LangParser::FunctionCallContext::LPAREN() {
  return getTokens(LangParser::LPAREN);
}

tree::TerminalNode* LangParser::FunctionCallContext::LPAREN(size_t i) {
  return getToken(LangParser::LPAREN, i);
}

std::vector<tree::TerminalNode *> LangParser::FunctionCallContext::RPAREN() {
  return getTokens(LangParser::RPAREN);
}

tree::TerminalNode* LangParser::FunctionCallContext::RPAREN(size_t i) {
  return getToken(LangParser::RPAREN, i);
}

std::vector<tree::TerminalNode *> LangParser::FunctionCallContext::DEC() {
  return getTokens(LangParser::DEC);
}

tree::TerminalNode* LangParser::FunctionCallContext::DEC(size_t i) {
  return getToken(LangParser::DEC, i);
}

std::vector<LangParser::ExpressionListContext *> LangParser::FunctionCallContext::expressionList() {
  return getRuleContexts<LangParser::ExpressionListContext>();
}

LangParser::ExpressionListContext* LangParser::FunctionCallContext::expressionList(size_t i) {
  return getRuleContext<LangParser::ExpressionListContext>(i);
}

LangParser::FunctionCallContext::FunctionCallContext(PostfixContext *ctx) { copyFrom(ctx); }

void LangParser::FunctionCallContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionCall(this);
}
void LangParser::FunctionCallContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionCall(this);
}

std::any LangParser::FunctionCallContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitFunctionCall(this);
  else
    return visitor->visitChildren(this);
}
LangParser::PostfixContext* LangParser::postfix() {
  PostfixContext *_localctx = _tracker.createInstance<PostfixContext>(_ctx, getState());
  enterRule(_localctx, 42, LangParser::RulePostfix);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    _localctx = _tracker.createInstance<LangParser::FunctionCallContext>(_localctx);
    enterOuterAlt(_localctx, 1);
    setState(211);
    primary();
    setState(220);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == LangParser::DEC

    || _la == LangParser::LPAREN) {
      setState(218);
      _errHandler->sync(this);
      switch (_input->LA(1)) {
        case LangParser::LPAREN: {
          setState(212);
          match(LangParser::LPAREN);
          setState(214);
          _errHandler->sync(this);

          _la = _input->LA(1);
          if ((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & 2203327395848) != 0)) {
            setState(213);
            expressionList();
          }
          setState(216);
          match(LangParser::RPAREN);
          break;
        }

        case LangParser::DEC: {
          setState(217);
          match(LangParser::DEC);
          break;
        }

      default:
        throw NoViableAltException(this);
      }
      setState(222);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- PrimaryContext ------------------------------------------------------------------

LangParser::PrimaryContext::PrimaryContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t LangParser::PrimaryContext::getRuleIndex() const {
  return LangParser::RulePrimary;
}

void LangParser::PrimaryContext::copyFrom(PrimaryContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- VariableContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::VariableContext::ID() {
  return getToken(LangParser::ID, 0);
}

LangParser::VariableContext::VariableContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::VariableContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVariable(this);
}
void LangParser::VariableContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVariable(this);
}

std::any LangParser::VariableContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitVariable(this);
  else
    return visitor->visitChildren(this);
}
//----------------- StringLiteralContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::StringLiteralContext::STRING() {
  return getToken(LangParser::STRING, 0);
}

LangParser::StringLiteralContext::StringLiteralContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::StringLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStringLiteral(this);
}
void LangParser::StringLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStringLiteral(this);
}

std::any LangParser::StringLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitStringLiteral(this);
  else
    return visitor->visitChildren(this);
}
//----------------- AnonymousFunctionContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::AnonymousFunctionContext::FN() {
  return getToken(LangParser::FN, 0);
}

tree::TerminalNode* LangParser::AnonymousFunctionContext::LPAREN() {
  return getToken(LangParser::LPAREN, 0);
}

tree::TerminalNode* LangParser::AnonymousFunctionContext::RPAREN() {
  return getToken(LangParser::RPAREN, 0);
}

tree::TerminalNode* LangParser::AnonymousFunctionContext::ARROW() {
  return getToken(LangParser::ARROW, 0);
}

LangParser::BlockStatementContext* LangParser::AnonymousFunctionContext::blockStatement() {
  return getRuleContext<LangParser::BlockStatementContext>(0);
}

LangParser::ParameterListContext* LangParser::AnonymousFunctionContext::parameterList() {
  return getRuleContext<LangParser::ParameterListContext>(0);
}

LangParser::AnonymousFunctionContext::AnonymousFunctionContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::AnonymousFunctionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnonymousFunction(this);
}
void LangParser::AnonymousFunctionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnonymousFunction(this);
}

std::any LangParser::AnonymousFunctionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitAnonymousFunction(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ParenthesizedContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::ParenthesizedContext::LPAREN() {
  return getToken(LangParser::LPAREN, 0);
}

LangParser::ExpressionContext* LangParser::ParenthesizedContext::expression() {
  return getRuleContext<LangParser::ExpressionContext>(0);
}

tree::TerminalNode* LangParser::ParenthesizedContext::RPAREN() {
  return getToken(LangParser::RPAREN, 0);
}

LangParser::ParenthesizedContext::ParenthesizedContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::ParenthesizedContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParenthesized(this);
}
void LangParser::ParenthesizedContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParenthesized(this);
}

std::any LangParser::ParenthesizedContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitParenthesized(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BoolFalseContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::BoolFalseContext::FALSE() {
  return getToken(LangParser::FALSE, 0);
}

LangParser::BoolFalseContext::BoolFalseContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::BoolFalseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBoolFalse(this);
}
void LangParser::BoolFalseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBoolFalse(this);
}

std::any LangParser::BoolFalseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitBoolFalse(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IntLiteralContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::IntLiteralContext::INT() {
  return getToken(LangParser::INT, 0);
}

LangParser::IntLiteralContext::IntLiteralContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::IntLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIntLiteral(this);
}
void LangParser::IntLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIntLiteral(this);
}

std::any LangParser::IntLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitIntLiteral(this);
  else
    return visitor->visitChildren(this);
}
//----------------- DoubleLiteralContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::DoubleLiteralContext::DOUBLE() {
  return getToken(LangParser::DOUBLE, 0);
}

LangParser::DoubleLiteralContext::DoubleLiteralContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::DoubleLiteralContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDoubleLiteral(this);
}
void LangParser::DoubleLiteralContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDoubleLiteral(this);
}

std::any LangParser::DoubleLiteralContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitDoubleLiteral(this);
  else
    return visitor->visitChildren(this);
}
//----------------- BoolTrueContext ------------------------------------------------------------------

tree::TerminalNode* LangParser::BoolTrueContext::TRUE() {
  return getToken(LangParser::TRUE, 0);
}

LangParser::BoolTrueContext::BoolTrueContext(PrimaryContext *ctx) { copyFrom(ctx); }

void LangParser::BoolTrueContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBoolTrue(this);
}
void LangParser::BoolTrueContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<LangListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBoolTrue(this);
}

std::any LangParser::BoolTrueContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<LangVisitor*>(visitor))
    return parserVisitor->visitBoolTrue(this);
  else
    return visitor->visitChildren(this);
}
LangParser::PrimaryContext* LangParser::primary() {
  PrimaryContext *_localctx = _tracker.createInstance<PrimaryContext>(_ctx, getState());
  enterRule(_localctx, 44, LangParser::RulePrimary);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(241);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case LangParser::INT: {
        _localctx = _tracker.createInstance<LangParser::IntLiteralContext>(_localctx);
        enterOuterAlt(_localctx, 1);
        setState(223);
        match(LangParser::INT);
        break;
      }

      case LangParser::DOUBLE: {
        _localctx = _tracker.createInstance<LangParser::DoubleLiteralContext>(_localctx);
        enterOuterAlt(_localctx, 2);
        setState(224);
        match(LangParser::DOUBLE);
        break;
      }

      case LangParser::STRING: {
        _localctx = _tracker.createInstance<LangParser::StringLiteralContext>(_localctx);
        enterOuterAlt(_localctx, 3);
        setState(225);
        match(LangParser::STRING);
        break;
      }

      case LangParser::TRUE: {
        _localctx = _tracker.createInstance<LangParser::BoolTrueContext>(_localctx);
        enterOuterAlt(_localctx, 4);
        setState(226);
        match(LangParser::TRUE);
        break;
      }

      case LangParser::FALSE: {
        _localctx = _tracker.createInstance<LangParser::BoolFalseContext>(_localctx);
        enterOuterAlt(_localctx, 5);
        setState(227);
        match(LangParser::FALSE);
        break;
      }

      case LangParser::ID: {
        _localctx = _tracker.createInstance<LangParser::VariableContext>(_localctx);
        enterOuterAlt(_localctx, 6);
        setState(228);
        match(LangParser::ID);
        break;
      }

      case LangParser::LPAREN: {
        _localctx = _tracker.createInstance<LangParser::ParenthesizedContext>(_localctx);
        enterOuterAlt(_localctx, 7);
        setState(229);
        match(LangParser::LPAREN);
        setState(230);
        expression();
        setState(231);
        match(LangParser::RPAREN);
        break;
      }

      case LangParser::FN: {
        _localctx = _tracker.createInstance<LangParser::AnonymousFunctionContext>(_localctx);
        enterOuterAlt(_localctx, 8);
        setState(233);
        match(LangParser::FN);
        setState(234);
        match(LangParser::LPAREN);
        setState(236);
        _errHandler->sync(this);

        _la = _input->LA(1);
        if (_la == LangParser::ID) {
          setState(235);
          parameterList();
        }
        setState(238);
        match(LangParser::RPAREN);
        setState(239);
        match(LangParser::ARROW);
        setState(240);
        blockStatement();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

void LangParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  langParserInitialize();
#else
  std::call_once(langParserOnceFlag, langParserInitialize);
#endif
}
