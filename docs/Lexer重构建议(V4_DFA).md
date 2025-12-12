# V4 Lexer DFA 重构建议

## 一、你的 Lexer 本质做了什么？

它识别这些正则集合：

| 类型   | 正则                     |
|--------|--------------------------|
| int    | `[0-9]+`                 |
| double | `[0-9]+ '.' [0-9]*`      |
| id     | `[a-zA-Z_][a-zA-Z0-9_]*` |
| string | `" [^"]* "`              |
| op2    | `== != <= >= &&`         |
| op1    | 单字符                   |
| keyword| id 的子集                |

这天然就是 DFA 的适用场景。

---

## 二、DFA 状态设计（核心）

我们定义状态：

| 编号 | 状态名          |
|------|----------------|
| 0    | START          |
| 1    | INT            |
| 2    | DOT_AFTER_INT  |
| 3    | DOUBLE         |
| 4    | ID             |
| 5    | STRING         |
| 6    | OP1_PREFIX     |
| 7    | DONE           |
| -1   | DEAD           |

接受状态：

| 状态 | token        |
|------|--------------|
| 1    | INT          |
| 3    | DOUBLE       |
| 4    | ID / KEYWORD |
| 5    | STRING       |
| 6    | OP           |

---

## 三、字符分类（减少状态表规模）

```cpp
enum CharClass {
    CC_DIGIT,
    CC_LETTER,
    CC_UNDER,
    CC_DOT,
    CC_QUOTE,
    CC_OP,
    CC_SPACE,
    CC_OTHER
};
```

分类函数：

```cpp
int classify(char c) {
    if (std::isdigit(c)) return CC_DIGIT;
    if (std::isalpha(c)) return CC_LETTER;
    if (c == '_') return CC_UNDER;
    if (c == '.') return CC_DOT;
    if (c == '"') return CC_QUOTE;
    if (std::isspace(c)) return CC_SPACE;
    if (std::string("+-*/%<>=!&|?:").find(c) != std::string::npos)
        return CC_OP;
    return CC_OTHER;
}
```

---

## 四、DFA 状态转移表

状态编号：

| 编号 | 状态   |
|------|--------|
| 0    | START  |
| 1    | INT    |
| 2    | DOT    |
| 3    | DOUBLE |
| 4    | ID     |
| 5    | STRING |
| 6    | OP     |

| state | digit | letter | _   | dot | "   | op  | space | other |
|-------|-------|--------|-----|-----|-----|-----|-------|-------|
| 0     | 1     | 4      | 4   | -1  | 5   | 6   | 0     | -1    |
| 1     | 1     | -1     | -1  | 2   | -1  | -1  | -1    | -1    |
| 2     | 3     | -1     | -1  | -1  | -1  | -1  | -1    | -1    |
| 3     | 3     | -1     | -1  | -1  | -1  | -1  | -1    | -1    |
| 4     | 4     | 4      | 4   | -1  | -1  | -1  | -1    | -1    |
| 5     | 5     | 5      | 5   | 5   | END | 5   | 5     | 5     |
| 6     | END   | END    | END | END | END | 6   | END   | END   |

（END = 停止扫描，回退一个字符）

---

## 五、DFA 驱动版完整实现

下面是工程级可直接替换的 `Lexer::nextToken()` 实现：

- 保留你的 Token 类型与关键字判断

### 代码（核心版）

```cpp
class Lexer {
    std::string src;
    size_t pos = 0;

    enum {
        S_START = 0,
        S_INT = 1,
        S_DOT = 2,
        S_DOUBLE = 3,
        S_ID = 4,
        S_STRING = 5,
        S_OP = 6
    };

    enum {
        CC_DIGIT,
        CC_LETTER,
        CC_UNDER,
        CC_DOT,
        CC_QUOTE,
        CC_OP,
        CC_SPACE,
        CC_OTHER
    };

    static int classify(char c) {
        if (std::isdigit(c)) return CC_DIGIT;
        if (std::isalpha(c)) return CC_LETTER;
        if (c == '_') return CC_UNDER;
        if (c == '.') return CC_DOT;
        if (c == '"') return CC_QUOTE;
        if (std::isspace(c)) return CC_SPACE;
        if (std::string("+-*/%<>=!&|?:").find(c) != std::string::npos)
            return CC_OP;
        return CC_OTHER;
    }

    static constexpr int DFA[7][8] = {
        {1,4,4,-1,5,6,0,-1},  // START
        {1,-1,-1,2,-1,-1,-1,-1}, // INT
        {3,-1,-1,-1,-1,-1,-1,-1}, // DOT
        {3,-1,-1,-1,-1,-1,-1,-1}, // DOUBLE
        {4,4,4,-1,-1,-1,-1,-1},   // ID
        {5,5,5,5,-2,5,5,5},       // STRING (-2 表示遇到结束引号)
        {-2,-2,-2,-2,-2,6,-2,-2}  // OP
    };

public:
    Lexer(const std::string& s) : src(s) {}

    Token nextToken() {
        while (pos < src.size() && std::isspace(src[pos])) pos++;

        if (pos >= src.size())
            return Token(TOK_EOF);

        size_t start = pos;
        int state = S_START;
        int lastAccept = -1;
        size_t lastPos = pos;

        while (pos < src.size()) {
            char c = src[pos];
            int cc = classify(c);
            int next = DFA[state][cc];

            if (next == -1) break;
            if (next == -2) { pos++; break; }

            state = next;
            pos++;

            if (state == S_INT || state == S_DOUBLE || state == S_ID || state == S_STRING || state == S_OP) {
                lastAccept = state;
                lastPos = pos;
            }
        }

        if (lastAccept == -1)
            throw std::runtime_error("Invalid token");

        pos = lastPos;
        std::string lexeme = src.substr(start, pos - start);

        // 分类 token
        if (lastAccept == S_INT)
            return Token(TOK_INT, lexeme);

        if (lastAccept == S_DOUBLE)
            return Token(TOK_DOUBLE, lexeme);

        if (lastAccept == S_STRING)
            return Token(TOK_STR, lexeme.substr(1, lexeme.size()-2));

        if (lastAccept == S_ID) {
            if (lexeme == "let") return Token(TOK_LET, lexeme);
            if (lexeme == "const") return Token(TOK_CONST, lexeme);
            if (lexeme == "fn") return Token(TOK_FN, lexeme);
            if (lexeme == "return") return Token(TOK_RETURN, lexeme);
            if (lexeme == "if") return Token(TOK_IF, lexeme);
            if (lexeme == "else") return Token(TOK_ELSE, lexeme);
            if (lexeme == "while") return Token(TOK_WHILE, lexeme);
            if (lexeme == "for") return Token(TOK_FOR, lexeme);
            if (lexeme == "break") return Token(TOK_BREAK, lexeme);
            if (lexeme == "continue") return Token(TOK_CONTINUE, lexeme);
            if (lexeme == "print") return Token(TOK_PRINT, lexeme);
            if (lexeme == "typeof") return Token(TOK_TYPEOF, lexeme);
            if (lexeme == "true") return Token(TOK_TRUE, lexeme);
            if (lexeme == "false") return Token(TOK_FALSE, lexeme);
            return Token(TOK_ID, lexeme);
        }

        if (lastAccept == S_OP) {
            // 先匹配双字符
            if (lexeme == "==") return Token(TOK_EQ, lexeme);
            if (lexeme == "!=") return Token(TOK_NEQ, lexeme);
            if (lexeme == "<=") return Token(TOK_LE, lexeme);
            if (lexeme == ">=") return Token(TOK_GE, lexeme);
            if (lexeme == "&&") return Token(TOK_AND, lexeme);
            if (lexeme == "||") return Token(TOK_OR, lexeme);
            if (lexeme == "+=") return Token(TOK_PLUS_ASSIGN, lexeme);
            if (lexeme == "-=") return Token(TOK_MINUS_ASSIGN, lexeme);
            if (lexeme == "*=") return Token(TOK_MUL_ASSIGN, lexeme);
            if (lexeme == "/=") return Token(TOK_DIV_ASSIGN, lexeme);
            if (lexeme == "=>") return Token(TOK_ARROW, lexeme);

            // 单字符 fallback
            char c = lexeme[0];
            switch (c) {
                case '+': return Token(TOK_PLUS, lexeme);
                case '-': return Token(TOK_MINUS, lexeme);
                case '*': return Token(TOK_MUL, lexeme);
                case '/': return Token(TOK_DIV, lexeme);
                case '%': return Token(TOK_MOD, lexeme);
                case '<': return Token(TOK_LT, lexeme);
                case '>': return Token(TOK_GT, lexeme);
                case '!': return Token(TOK_NOT, lexeme);
                case '=': return Token(TOK_ASSIGN, lexeme);
                case '(': return Token(TOK_LP, lexeme);
                case ')': return Token(TOK_RP, lexeme);
                case '{': return Token(TOK_LB, lexeme);
                case '}': return Token(TOK_RB, lexeme);
                case ';': return Token(TOK_SEMI, lexeme);
                case ',': return Token(TOK_COMMA, lexeme);
                case '?': return Token(TOK_QUESTION, lexeme);
                case ':': return Token(TOK_COLON, lexeme);
            }
        }

        throw std::runtime_error("Unknown token");
    }
};
```
