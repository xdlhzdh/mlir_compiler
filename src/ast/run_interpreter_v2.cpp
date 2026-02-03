#include <iostream>
#include <string>

#include "interpreter_v2.hpp"

using namespace Interpreter_V2;

int main() {
  std::cout << "===== Compiler V2 Demo =====" << std::endl;

  std::string source = R"(
fn fact(n) {
	if (n == 0) { return 1; }
	return n * fact(n - 1);
}

fn add(a, b) {
	return a + b;
}

let x = 5;
print("fact(5) =", fact(5));
print("add(3,4) =", add(3,4));

let s = "hello";
let t = " world";
print(s + t);

let b = true;
if (b && (1 < 2)) { print("branch true"); } else { print("branch false"); }

let i = 3;
while (i > 0) {
	print("i:", i);
	i = i - 1;
}
)";

  src = source;
  pos = 0;
  getNext();

  std::vector<Stmt *> program;
  while (curTok.type != TOK_EOF) {
    if (curTok.type == TOK_FN)
      parseFunction();
    else
      program.push_back(parseStatement());
  }

  Env env;
  for (auto s : program)
    s->exec(env);

  return 0;
}
