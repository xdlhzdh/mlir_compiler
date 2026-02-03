#include <iostream>
#include <map>
#include <string>

#include "interpreter_v1.hpp"

using namespace Interpreter_V1;

int main() {
  std::cout << "===== Compiler V1 Demo =====" << std::endl;

  std::string src = "x = 1 + 2 * 3;"
                    "print(x);"
                    "x = (x + 10) / 2;"
                    "print(x);";

  Lexer lex(src);
  Parser parser(lex);

  auto program = parser.parseProgram();

  std::map<std::string, int> ctx; // Variable context

  for (auto &stmt : program)
    stmt->exec(ctx);

  return 0;
}
