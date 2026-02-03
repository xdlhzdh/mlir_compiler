#include "interpreter_v4.hpp"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdlib>
#include <iostream>

using namespace Interpreter_V4;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << (argv[0] ? argv[0] : "demo_interpreter_v4_llvm")
              << " <output.ll>\n";
    return EXIT_FAILURE;
  }
  const char *output_path = argv[1];

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

  std::error_code EC;
  llvm::raw_fd_ostream out(output_path, EC, llvm::sys::fs::OF_None);
  if (EC) {
    std::cerr << "Error opening " << output_path << ": " << EC.message()
              << "\n";
    return EXIT_FAILURE;
  }
  module->print(out, nullptr);
  out.close();
  return 0;
}
