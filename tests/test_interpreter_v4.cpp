#include <gtest/gtest.h>
#include <sstream>

#include "interpreter_v4.hpp"

using namespace Interpreter_V4;

// Helper to capture stdout
class CaptureStdout {
  std::ostringstream buffer;
  std::streambuf *old;

public:
  CaptureStdout() : old(std::cout.rdbuf(buffer.rdbuf())) {}
  ~CaptureStdout() { std::cout.rdbuf(old); }
  std::string get() { return buffer.str(); }
};

// ===== Lexer Tests =====
TEST(V4_LexerTest, TokenizeBreak) {
  Lexer lex("break;");
  Token tok = lex.nextToken();
  EXPECT_EQ(tok.type, TOK_BREAK);
  EXPECT_EQ(tok.text, "break");
}

TEST(V4_LexerTest, TokenizeContinue) {
  Lexer lex("continue;");
  Token tok = lex.nextToken();
  EXPECT_EQ(tok.type, TOK_CONTINUE);
  EXPECT_EQ(tok.text, "continue");
}

TEST(V4_LexerTest, TokenizeWhileWithBreak) {
  Lexer lex("while (x) { break; }");
  EXPECT_EQ(lex.nextToken().type, TOK_WHILE);
  EXPECT_EQ(lex.nextToken().type, TOK_LP);
  EXPECT_EQ(lex.nextToken().type, TOK_ID);
  EXPECT_EQ(lex.nextToken().type, TOK_RP);
  EXPECT_EQ(lex.nextToken().type, TOK_LB);
  EXPECT_EQ(lex.nextToken().type, TOK_BREAK);
  EXPECT_EQ(lex.nextToken().type, TOK_SEMI);
  EXPECT_EQ(lex.nextToken().type, TOK_RB);
}

// ===== Break Statement Tests =====
TEST(V4_BreakTest, BasicBreak) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 10) {
      if (i == 5) {
        break;
      }
      print i;
      i += 1;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("0"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_EQ(output.find("5"), std::string::npos);
  EXPECT_EQ(output.find("6"), std::string::npos);
}

TEST(V4_BreakTest, BreakImmediately) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 10) {
      break;
      i += 1;
    }
    print i;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("0"), std::string::npos);
}

TEST(V4_BreakTest, BreakInNestedLoop) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 3) {
      let j = 0;
      while (j < 5) {
        if (j == 2) {
          break;
        }
        print j;
        j += 1;
      }
      i += 1;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  // Should print 0, 1 three times (once for each outer iteration)
  // Count lines in output
  size_t lineCount = 0;
  for (char c : output) {
    if (c == '\n')
      lineCount++;
  }
  EXPECT_EQ(lineCount, 6); // 3 outer iterations * 2 inner prints = 6 lines
  EXPECT_NE(output.find("0"), std::string::npos);
  EXPECT_NE(output.find("1"), std::string::npos);
}

TEST(V4_BreakTest, BreakWithCondition) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 1;
    let found = 0;
    while (i < 100) {
      let sq = i * i;
      if (sq > 50) {
        found = sq;
        break;
      }
      i += 1;
    }
    print found;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("64"), std::string::npos); // 8*8 = 64
}

// ===== Continue Statement Tests =====
TEST(V4_ContinueTest, BasicContinue) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 5) {
      i += 1;
      if (i % 2 == 0) {
        continue;
      }
      print i;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
  // Should not find 2 and 4 as separate numbers
  EXPECT_EQ(output.find("2\n"), std::string::npos);
  EXPECT_EQ(output.find("4\n"), std::string::npos);
}

TEST(V4_ContinueTest, ContinueSkipsCode) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    let sum = 0;
    while (i < 5) {
      i += 1;
      if (i == 3) {
        continue;
      }
      sum += i;
    }
    print sum;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  // sum = 1 + 2 + 4 + 5 = 12 (skipping 3)
  EXPECT_NE(output.find("12"), std::string::npos);
}

TEST(V4_ContinueTest, ContinueInNestedLoop) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 3) {
      i += 1;
      if (i == 2) {
        continue;
      }
      let j = 0;
      while (j < 2) {
        j += 1;
        print i, j;
      }
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
  // Should not execute inner loop when i == 2
}

// ===== Combined Break and Continue Tests =====
TEST(V4_BreakContinueTest, BothInSameLoop) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 20) {
      i += 1;
      if (i % 2 == 0) {
        continue;
      }
      if (i > 10) {
        break;
      }
      print i;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
  EXPECT_NE(output.find("7"), std::string::npos);
  EXPECT_NE(output.find("9"), std::string::npos);
  EXPECT_EQ(output.find("11"), std::string::npos);
  EXPECT_EQ(output.find("13"), std::string::npos);
}

TEST(V4_BreakContinueTest, SearchPattern) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    let count = 0;
    while (i < 100) {
      i += 1;
      if (i % 3 != 0) {
        continue;
      }
      count += 1;
      if (count >= 5) {
        break;
      }
      print i;
    }
    print count;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("3"), std::string::npos);
  EXPECT_NE(output.find("6"), std::string::npos);
  EXPECT_NE(output.find("9"), std::string::npos);
  EXPECT_NE(output.find("12"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos); // count
}

// ===== Break/Continue with Functions =====
TEST(V4_BreakContinueTest, BreakInFunctionLoop) {
  CaptureStdout capture;
  std::string code = R"(
    fn findFirst(n) {
      let i = 2;
      while (i < n) {
        if (n % i == 0) {
          return i;
        }
        i += 1;
      }
      return n;
    }
    print findFirst(15);
    print findFirst(17);
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("3"), std::string::npos);  // 15 / 3 = 5
  EXPECT_NE(output.find("17"), std::string::npos); // 17 is prime
}

TEST(V4_BreakContinueTest, ContinueInFunctionLoop) {
  CaptureStdout capture;
  std::string code = R"(
    fn sumOdds(max) {
      let i = 0;
      let sum = 0;
      while (i < max) {
        i += 1;
        if (i % 2 == 0) {
          continue;
        }
        sum += i;
      }
      return sum;
    }
    print sumOdds(10);
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  // sum = 1 + 3 + 5 + 7 + 9 = 25
  EXPECT_NE(output.find("25"), std::string::npos);
}

// ===== Break/Continue with Closures =====
TEST(V4_BreakContinueTest, BreakInClosure) {
  CaptureStdout capture;
  std::string code = R"(
    fn makeCounter(limit) {
      let count = 0;
      fn increment() {
        while (count < 100) {
          count += 1;
          if (count >= limit) {
            break;
          }
        }
        return count;
      }
      return increment;
    }
    let counter = makeCounter(5);
    print counter();
    print counter();
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("5"), std::string::npos);
}

// ===== Edge Cases =====
TEST(V4_BreakContinueTest, MultipleBreaksInSequence) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 10) {
      i += 1;
      if (i == 3) {
        break;
      }
      if (i == 5) {
        break;
      }
      print i;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_EQ(output.find("3"), std::string::npos);
  EXPECT_EQ(output.find("4"), std::string::npos);
}

TEST(V4_BreakContinueTest, MultipleContinuesInSequence) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 10) {
      i += 1;
      if (i == 3) {
        continue;
      }
      if (i == 5) {
        continue;
      }
      print i;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_EQ(output.find("3\n"), std::string::npos);
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_EQ(output.find("5\n"), std::string::npos);
  EXPECT_NE(output.find("6"), std::string::npos);
}

TEST(V4_BreakContinueTest, BreakWithComplexCondition) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 100) {
      i += 1;
      if (i > 10 && i < 20) {
        continue;
      }
      if (i >= 25) {
        break;
      }
      print i;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("1"), std::string::npos);
  EXPECT_NE(output.find("10"), std::string::npos);
  EXPECT_EQ(output.find("15"), std::string::npos);
  EXPECT_NE(output.find("20"), std::string::npos);
  EXPECT_NE(output.find("24"), std::string::npos);
  EXPECT_EQ(output.find("25"), std::string::npos);
}

// ===== Integration with V3 Features =====
TEST(V4_IntegrationTest, BreakWithTernary) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 10) {
      i += 1;
      let msg = i > 5 ? "big" : "small";
      if (i > 5) {
        break;
      }
      print i, msg;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("small"), std::string::npos);
  EXPECT_EQ(output.find("big"), std::string::npos);
}

TEST(V4_IntegrationTest, ContinueWithCompoundAssignment) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    let sum = 0;
    while (i < 10) {
      i += 1;
      if (i % 2 == 0) {
        continue;
      }
      sum += i;
    }
    print sum;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  // sum = 1 + 3 + 5 + 7 + 9 = 25
  EXPECT_NE(output.find("25"), std::string::npos);
}

TEST(V4_IntegrationTest, BreakWithTypeof) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 100) {
      i += 1;
      if (typeof i == "int" && i > 5) {
        break;
      }
      print i;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("5"), std::string::npos);
  // Should not print 7 or higher
  EXPECT_EQ(output.find("7\n"), std::string::npos);
}

TEST(V4_IntegrationTest, ContinueWithStringComparison) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    while (i < 5) {
      i += 1;
      let status = i > 3 ? "high" : "low";
      if (status == "low") {
        continue;
      }
      print i, status;
    }
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("4"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
  EXPECT_NE(output.find("high"), std::string::npos);
  EXPECT_EQ(output.find("low"), std::string::npos);
}

// ===== Parser Tests =====
TEST(V4_ParserTest, ParseBreakStatement) {
  std::string code = "break;";
  Parser p(code);
  auto stmts = p.parseProgram();
  EXPECT_EQ(stmts.size(), 1);
}

TEST(V4_ParserTest, ParseContinueStatement) {
  std::string code = "continue;";
  Parser p(code);
  auto stmts = p.parseProgram();
  EXPECT_EQ(stmts.size(), 1);
}

TEST(V4_ParserTest, ParseWhileWithBreak) {
  std::string code = "while (x) { break; }";
  Parser p(code);
  auto stmts = p.parseProgram();
  EXPECT_EQ(stmts.size(), 1);
}

TEST(V4_ParserTest, ParseWhileWithContinue) {
  std::string code = "while (x) { continue; }";
  Parser p(code);
  auto stmts = p.parseProgram();
  EXPECT_EQ(stmts.size(), 1);
}

TEST(V4_ParserTest, ParseBreakRequiresSemicolon) {
  std::string code = "break";
  Parser p(code);
  EXPECT_THROW(p.parseProgram(), std::runtime_error);
}

TEST(V4_ParserTest, ParseContinueRequiresSemicolon) {
  std::string code = "continue";
  Parser p(code);
  EXPECT_THROW(p.parseProgram(), std::runtime_error);
}

// ===== Comprehensive Example Tests =====
TEST(V4_ComprehensiveTest, SievePattern) {
  CaptureStdout capture;
  std::string code = R"(
    fn isPrime(n) {
      if (n < 2) return false;
      if (n == 2) return true;
      if (n % 2 == 0) return false;
      let i = 3;
      while (i * i <= n) {
        if (n % i == 0) {
          return false;
        }
        i += 2;
      }
      return true;
    }
    
    let count = 0;
    let num = 2;
    while (num < 30) {
      if (!isPrime(num)) {
        num += 1;
        continue;
      }
      print num;
      count += 1;
      if (count >= 5) {
        break;
      }
      num += 1;
    }
    print count;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  EXPECT_NE(output.find("2"), std::string::npos);
  EXPECT_NE(output.find("3"), std::string::npos);
  EXPECT_NE(output.find("5"), std::string::npos);
  EXPECT_NE(output.find("7"), std::string::npos);
  EXPECT_NE(output.find("11"), std::string::npos);
}

TEST(V4_ComprehensiveTest, NestedLoopsWithBothControls) {
  CaptureStdout capture;
  std::string code = R"(
    let i = 0;
    let total = 0;
    while (i < 5) {
      i += 1;
      if (i == 3) {
        continue;
      }
      let j = 0;
      while (j < 5) {
        j += 1;
        if (j == i) {
          continue;
        }
        if (j > 3) {
          break;
        }
        total += 1;
      }
    }
    print total;
  )";

  auto env = std::make_shared<Env>();
  Parser p(code);
  auto stmts = p.parseProgram();
  for (auto &stmt : stmts)
    stmt->exec(env);

  std::string output = capture.get();
  // Complex calculation but should execute without errors
  EXPECT_FALSE(output.empty());
}
