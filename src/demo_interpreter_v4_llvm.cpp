#include "interpreter_v4.hpp"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

using namespace Interpreter_V4;

int main() {
  const char *code = R"(
fn add(a, b) {
  return a + b;
}

fn fib(n) {
  if (n < 2) return n;
  return fib(n - 1) + fib(n - 2);
}

let x = 3;
let y = 4;
print add(x, y);
print fib(5);
)";

  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      std::make_unique<llvm::Module>("interpreter_v4_ir", context);
  llvm::IRBuilder<> builder(context);

  Parser p(code);
  auto program = p.parseProgram();
  generateProgram(module.get(), context, builder, program);

  module->print(llvm::outs(), nullptr);
  return 0;
}
