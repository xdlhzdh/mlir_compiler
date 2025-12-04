#pragma once

#include <bits/stdc++.h>

namespace Interpreter_V2 {

// ===== Lexer =====
enum TokenType {
  TOK_EOF,    // End of file
  TOK_ID,     // Identifier (e.g., variable or function name)
  TOK_NUM,    // Numeric literal
  TOK_STR,    // String literal
  TOK_LET,    // 'let' keyword for variable declaration
  TOK_FN,     // 'fn' keyword for function definition
  TOK_RETURN, // 'return' keyword for returning a value
  TOK_IF,     // 'if' keyword for conditional statements
  TOK_ELSE,   // 'else' keyword for alternative branch in conditionals
  TOK_WHILE,  // 'while' keyword for loops
  TOK_PRINT,  // 'print' keyword for output
  TOK_OP,     // Operator (e.g., +, -, *, /, etc.)
  TOK_EQ,     // Equality operator (==)
  TOK_NEQ,    // Inequality operator (!=)
  TOK_ASSIGN, // Assignment operator (=)
  TOK_LP,     // Left parenthesis '('
  TOK_RP,     // Right parenthesis ')'
  TOK_LB,     // Left brace '{'
  TOK_RB,     // Right brace '}'
  TOK_SEMI,   // Semicolon ';'
  TOK_COMMA   // Comma ','
};

struct Token {
  TokenType type;
  std::string text;
  Token(TokenType t = TOK_EOF, std::string s = "")
      : type(t), text(std::move(s)) {}
};

std::string src;
int pos = 0;

bool isIdStart(char c) { return std::isalpha(c) || c == '_'; }
bool isIdChar(char c) { return std::isalnum(c) || c == '_'; }

Token nextToken() {
  while (pos < (int)src.size() && std::isspace(src[pos]))
    pos++;
  if (pos >= (int)src.size())
    return Token(TOK_EOF, "");

  char c = src[pos];

  // number
  if (std::isdigit(c)) {
    int st = pos;
    while (pos < (int)src.size() && std::isdigit(src[pos]))
      pos++;
    return Token(TOK_NUM, src.substr(st, pos - st));
  }

  // identifier or keyword
  if (isIdStart(c)) {
    int st = pos;
    while (pos < (int)src.size() && isIdChar(src[pos]))
      pos++;
    std::string w = src.substr(st, pos - st);
    if (w == "let")
      return Token(TOK_LET, w);
    if (w == "fn")
      return Token(TOK_FN, w);
    if (w == "return")
      return Token(TOK_RETURN, w);
    if (w == "if")
      return Token(TOK_IF, w);
    if (w == "else")
      return Token(TOK_ELSE, w);
    if (w == "while")
      return Token(TOK_WHILE, w);
    if (w == "print")
      return Token(TOK_PRINT, w);
    if (w == "true")
      return Token(TOK_NUM, "1");
    if (w == "false")
      return Token(TOK_NUM, "0");
    return Token(TOK_ID, w);
  }

  // string literal
  if (c == '"') {
    pos++;
    int st = pos;
    while (pos < (int)src.size() && src[pos] != '"')
      pos++;
    std::string s = src.substr(st, pos - st);
    pos++; // eat closing
    return Token(TOK_STR, s);
  }

  // operators
  if (c == '=') {
    if (pos + 1 < (int)src.size() && src[pos + 1] == '=') {
      pos += 2;
      return Token(TOK_EQ, "==");
    }
    pos++;
    return Token(TOK_ASSIGN, "=");
  }
  if (c == '!') {
    if (pos + 1 < (int)src.size() && src[pos + 1] == '=') {
      pos += 2;
      return Token(TOK_NEQ, "!=");
    }
  }
  if (c == '<' || c == '>') {
    char op = c;
    pos++;
    if (pos < (int)src.size() && src[pos] == '=') {
      std::string s;
      s += op;
      s += '=';
      pos++;
      return Token(TOK_OP, s);
    }
    return Token(TOK_OP, std::string(1, op));
  }
  if (c == '&' && pos + 1 < (int)src.size() && src[pos + 1] == '&') {
    pos += 2;
    return Token(TOK_OP, "&&");
  }
  if (c == '|' && pos + 1 < (int)src.size() && src[pos + 1] == '|') {
    pos += 2;
    return Token(TOK_OP, "||");
  }

  if (c == '+' || c == '-' || c == '*' || c == '/') {
    pos++;
    return Token(TOK_OP, std::string(1, c));
  }

  if (c == '(') {
    pos++;
    return Token(TOK_LP, "(");
  }
  if (c == ')') {
    pos++;
    return Token(TOK_RP, ")");
  }
  if (c == '{') {
    pos++;
    return Token(TOK_LB, "{");
  }
  if (c == '}') {
    pos++;
    return Token(TOK_RB, "}");
  }
  if (c == ';') {
    pos++;
    return Token(TOK_SEMI, ";");
  }
  if (c == ',') {
    pos++;
    return Token(TOK_COMMA, ",");
  }

  pos++;
  return Token(TOK_EOF, "?");
}

Token curTok;
Token getNext() {
  curTok = nextToken();
  return curTok;
}

// ===== AST value =====
struct Value {
  int i = 0;
  std::string s;
  bool isStr = false;
  Value(int v = 0) : i(v), isStr(false) {}
  Value(std::string v) : s(std::move(v)), isStr(true) {}
};

// ===== Environment =====
struct Env {
  std::vector<std::unordered_map<std::string, Value>> st;
  Env() { st.emplace_back(); }
  void push() { st.emplace_back(); }
  void pop() { st.pop_back(); }

  bool has(const std::string &k) {
    for (int i = st.size() - 1; i >= 0; --i)
      if (st[i].count(k))
        return true;
    return false;
  }

  Value &get(const std::string &k) {
    for (int i = st.size() - 1; i >= 0; --i)
      if (st[i].count(k))
        return st[i][k];
    throw std::runtime_error("Undefined variable: " + k);
  }

  void setLocal(const std::string &k, Value v) { st.back()[k] = v; }
};

// ===== AST nodes =====
struct Expr {
  virtual Value eval(Env &) = 0;
  virtual ~Expr() {}
};

struct NumExpr : Expr {
  int v;
  NumExpr(int x) : v(x) {}
  Value eval(Env &) { return Value(v); }
};
struct StrExpr : Expr {
  std::string v;
  StrExpr(std::string x) : v(std::move(x)) {}
  Value eval(Env &) { return Value(v); }
};
struct VarExpr : Expr {
  std::string name;
  VarExpr(std::string n) : name(std::move(n)) {}
  Value eval(Env &env) { return env.get(name); }
};

struct BinaryExpr : Expr {
  std::string op;
  Expr *l;
  Expr *r;
  BinaryExpr(std::string o, Expr *a, Expr *b) : op(std::move(o)), l(a), r(b) {}
  Value eval(Env &env) {
    Value A = l->eval(env);
    Value B = r->eval(env);
    if (op == "+") {
      if (A.isStr || B.isStr) {
        std::string left = A.isStr ? A.s : std::to_string(A.i);
        std::string right = B.isStr ? B.s : std::to_string(B.i);
        return Value(left + right);
      }
      return Value(A.i + B.i);
    }
    if (op == "-")
      return Value(A.i - B.i);
    if (op == "*")
      return Value(A.i * B.i);
    if (op == "/")
      return Value(A.i / B.i);
    if (op == "==")
      return Value(A.i == B.i);
    if (op == "!=")
      return Value(A.i != B.i);
    if (op == "<")
      return Value(A.i < B.i);
    if (op == ">")
      return Value(A.i > B.i);
    if (op == "<=")
      return Value(A.i <= B.i);
    if (op == ">=")
      return Value(A.i >= B.i);
    if (op == "&&")
      return Value(A.i && B.i);
    if (op == "||")
      return Value(A.i || B.i);
    return Value(0);
  }
};

struct CallExpr : Expr {
  std::string name;
  std::vector<Expr *> args;
  CallExpr(std::string n, std::vector<Expr *> a)
      : name(std::move(n)), args(std::move(a)) {}
  Value eval(Env &env);
};

// ===== Statements =====
struct Stmt {
  virtual void exec(Env &) = 0;
  virtual ~Stmt() {}
};
struct BlockStmt : Stmt {
  std::vector<Stmt *> body;
  void exec(Env &env) {
    env.push();
    for (auto s : body)
      s->exec(env);
    env.pop();
  }
};
struct LetStmt : Stmt {
  std::string name;
  Expr *e;
  LetStmt(std::string n, Expr *x) : name(std::move(n)), e(x) {}
  void exec(Env &env) { env.setLocal(name, e->eval(env)); }
};
struct AssignStmt : Stmt {
  std::string name;
  Expr *e;
  AssignStmt(std::string n, Expr *x) : name(std::move(n)), e(x) {}
  void exec(Env &env) {
    if (!env.has(name))
      throw std::runtime_error("Undefined variable: " + name);
    env.get(name) = e->eval(env);
  }
};
struct ExprStmt : Stmt {
  Expr *e;
  ExprStmt(Expr *x) : e(x) {}
  void exec(Env &env) { e->eval(env); }
};
struct PrintStmt : Stmt {
  std::vector<Expr *> exprs;
  PrintStmt(std::vector<Expr *> e) : exprs(std::move(e)) {}
  void exec(Env &env) {
    for (int i = 0; i < (int)exprs.size(); i++) {
      Value v = exprs[i]->eval(env);
      if (v.isStr)
        std::cout << v.s;
      else
        std::cout << v.i;
      if (i != (int)exprs.size() - 1)
        std::cout << " ";
    }
    std::cout << "\n";
  }
};
struct IfStmt : Stmt {
  Expr *c;
  Stmt *t;
  Stmt *f;
  IfStmt(Expr *a, Stmt *b, Stmt *c2) : c(a), t(b), f(c2) {}
  void exec(Env &env) {
    if (c->eval(env).i)
      t->exec(env);
    else if (f)
      f->exec(env);
  }
};
struct WhileStmt : Stmt {
  Expr *c;
  Stmt *b;
  WhileStmt(Expr *a, Stmt *bb) : c(a), b(bb) {}
  void exec(Env &env) {
    while (c->eval(env).i)
      b->exec(env);
  }
};
struct ReturnExc {
  Value v;
  ReturnExc(Value x) : v(x) {};
};
struct ReturnStmt : Stmt {
  Expr *e;
  ReturnStmt(Expr *x) : e(x) {}
  void exec(Env &env) {
    Value v = e ? e->eval(env) : Value(0);
    throw ReturnExc(v);
  }
};

// function table
struct FuncDef {
  std::vector<std::string> params;
  Stmt *body;
};
std::unordered_map<std::string, FuncDef *> funcTable;

Value CallExpr::eval(Env &env) {
  auto it = funcTable.find(name);
  if (it == funcTable.end())
    throw std::runtime_error("Undefined function: " + name);
  FuncDef *f = it->second;
  if (f->params.size() != args.size())
    throw std::runtime_error("arg count mismatch");
  Env local = env;
  local.push();
  for (int i = 0; i < (int)args.size(); ++i)
    local.setLocal(f->params[i], args[i]->eval(env));
  try {
    f->body->exec(local);
    return Value(0);
  } catch (ReturnExc &re) {
    return re.v;
  }
}

// ===== Parser =====
Expr *parseExpression();
Stmt *parseStatement();
Stmt *parseBlock();
bool accept(TokenType t) {
  if (curTok.type == t) {
    getNext();
    return true;
  }
  return false;
}
void expect(TokenType t) {
  if (curTok.type != t) {
    std::string msg = "unexpected token: got '" + curTok.text +
                      "' (type=" + std::to_string((int)curTok.type) +
                      "), expected type=" + std::to_string((int)t);
    throw std::runtime_error(msg);
  }
  getNext();
}

Expr *parsePrimary() {
  if (curTok.type == TOK_NUM) {
    int v = std::stoi(curTok.text);
    getNext();
    return new NumExpr(v);
  }
  if (curTok.type == TOK_STR) {
    std::string v = curTok.text;
    getNext();
    return new StrExpr(v);
  }
  if (curTok.type == TOK_ID) {
    std::string id = curTok.text;
    getNext();
    if (curTok.type == TOK_LP) {
      getNext();
      std::vector<Expr *> a;
      if (curTok.type != TOK_RP) {
        a.push_back(parseExpression());
        while (accept(TOK_COMMA))
          a.push_back(parseExpression());
      }
      expect(TOK_RP);
      return new CallExpr(id, a);
    }
    return new VarExpr(id);
  }
  if (accept(TOK_LP)) {
    Expr *e = parseExpression();
    expect(TOK_RP);
    return e;
  }
  throw std::runtime_error("bad primary");
}

int prec(const std::string &op) {
  if (op == "||")
    return 1;
  if (op == "&&")
    return 2;
  if (op == "==" || op == "!=" || op == "<" || op == ">" || op == "<=" ||
      op == ">=")
    return 3;
  if (op == "+" || op == "-")
    return 4;
  if (op == "*" || op == "/")
    return 5;
  return -1;
}

Expr *parseBinOpRHS(int p, Expr *lhs) {
  while (curTok.type == TOK_OP || curTok.type == TOK_EQ ||
         curTok.type == TOK_NEQ) {
    std::string op = curTok.text;
    int opPrec = prec(op);
    if (opPrec < p)
      break;
    getNext();
    Expr *rhs = parsePrimary();
    while (curTok.type == TOK_OP || curTok.type == TOK_EQ ||
           curTok.type == TOK_NEQ) {
      std::string op2 = curTok.text;
      int prec2 = prec(op2);
      if (prec2 <= opPrec)
        break;
      rhs = parseBinOpRHS(prec2, rhs);
    }
    lhs = new BinaryExpr(op, lhs, rhs);
  }
  return lhs;
}

Expr *parseExpression() { return parseBinOpRHS(0, parsePrimary()); }

Stmt *parseStatement() {
  if (accept(TOK_LET)) {
    if (curTok.type != TOK_ID)
      throw std::runtime_error("expected identifier after let");
    std::string id = curTok.text;
    getNext();
    expect(TOK_ASSIGN);
    Expr *e = parseExpression();
    expect(TOK_SEMI);
    return new LetStmt(id, e);
  }
  if (accept(TOK_PRINT)) {
    bool hasParen = accept(TOK_LP);
    std::vector<Expr *> args;
    args.push_back(parseExpression());
    while (accept(TOK_COMMA))
      args.push_back(parseExpression());
    if (hasParen)
      expect(TOK_RP);
    expect(TOK_SEMI);
    return new PrintStmt(args);
  }
  if (accept(TOK_IF)) {
    expect(TOK_LP);
    Expr *c = parseExpression();
    expect(TOK_RP);
    Stmt *t = parseStatement();
    Stmt *f = nullptr;
    if (accept(TOK_ELSE))
      f = parseStatement();
    return new IfStmt(c, t, f);
  }
  if (accept(TOK_WHILE)) {
    expect(TOK_LP);
    Expr *c = parseExpression();
    expect(TOK_RP);
    Stmt *b = parseStatement();
    return new WhileStmt(c, b);
  }
  if (accept(TOK_RETURN)) {
    Expr *e = nullptr;
    if (curTok.type != TOK_SEMI)
      e = parseExpression();
    expect(TOK_SEMI);
    return new ReturnStmt(e);
  }
  if (curTok.type == TOK_LB)
    return parseBlock();

  // Check for assignment: ID = expr;
  if (curTok.type == TOK_ID) {
    std::string id = curTok.text;
    getNext();
    if (curTok.type == TOK_ASSIGN) {
      getNext();
      Expr *e = parseExpression();
      expect(TOK_SEMI);
      return new AssignStmt(id, e);
    }
    // Not an assignment, backtrack by re-parsing as expression
    // This is a limitation - we consumed the ID already
    // We need to create a VarExpr and continue parsing
    Expr *lhs = new VarExpr(id);
    // Check if there's an operator following
    if (curTok.type == TOK_OP || curTok.type == TOK_EQ ||
        curTok.type == TOK_NEQ) {
      lhs = parseBinOpRHS(0, lhs);
    }
    // Could also be a function call, but we already consumed the ID
    // So this is a simplified approach
    expect(TOK_SEMI);
    return new ExprStmt(lhs);
  }

  Expr *e = parseExpression();
  expect(TOK_SEMI);
  return new ExprStmt(e);
}

Stmt *parseBlock() {
  expect(TOK_LB);
  BlockStmt *b = new BlockStmt();
  while (curTok.type != TOK_RB)
    b->body.push_back(parseStatement());
  expect(TOK_RB);
  return b;
}

void parseFunction() {
  expect(TOK_FN);
  if (curTok.type != TOK_ID)
    throw std::runtime_error("expected function name");
  std::string name = curTok.text;
  getNext();
  expect(TOK_LP);
  std::vector<std::string> params;
  if (curTok.type != TOK_RP) {
    do {
      if (curTok.type != TOK_ID)
        throw std::runtime_error("expected parameter name");
      params.push_back(curTok.text);
      getNext();
    } while (accept(TOK_COMMA));
  }
  expect(TOK_RP);
  Stmt *body = parseBlock();
  funcTable[name] = new FuncDef{params, body};
}

} // namespace Interpreter_V2
