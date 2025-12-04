#include "simple_llvm_ir.hpp"

int main() {
  // 1. Create LLVMContext: foundation of LLVM objects
  LLVMContext TheContext;

  // 2. Create Module: top-level container
  std::unique_ptr<Module> TheModule =
      std::make_unique<Module>("simple_llvm_ir", TheContext);

  // 3. Build IR
  simple_llvm_ir::createAddFunction(*TheModule, TheContext);

  // 4. Print generated IR to stdout
  // Can also use TheModule->print(OutputFile, nullptr); to output to file
  TheModule->print(errs(), nullptr);

  return 0;
}