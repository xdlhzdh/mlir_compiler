#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/TargetSelect.h>

#include <iostream>
#include <memory>

#include "simple_llvm_ir.hpp"

using namespace llvm;
using namespace llvm::orc;

int main() {
  // **1. Initialize JIT/Target**
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  auto TheContext = std::make_unique<LLVMContext>();
  std::unique_ptr<Module> TheModule =
      std::make_unique<Module>("my_jit_module", *TheContext);

  // **2. Create IR**
  simple_llvm_ir::createAddFunction(*TheModule, *TheContext);

  // **3. Create LLJIT instance**
  auto JITResult = LLJITBuilder().create();
  if (!JITResult) {
    std::cerr << "Failed to create LLJIT: " << toString(JITResult.takeError())
              << std::endl;
    return 1;
  }
  auto JIT = std::move(*JITResult);

  // **4. Add Module to JIT**
  if (auto Err = JIT->addIRModule(
          ThreadSafeModule(std::move(TheModule), std::move(TheContext)))) {
    std::cerr << "Failed to add module to JIT: " << toString(std::move(Err))
              << std::endl;
    return 1;
  }

  // **5. Find compiled function address**
  auto AddSym = JIT->lookup("add");
  if (!AddSym) {
    std::cerr << "Failed to find 'add' function: "
              << toString(AddSym.takeError()) << std::endl;
    return 1;
  }

  using AddFuncPtr = int (*)(int, int);
  AddFuncPtr Add = AddSym->toPtr<AddFuncPtr>();

  // **6. Execute compiled code**
  int result = Add(2, 3);
  std::cout << "JIT Execution Result (2 + 3): " << result << std::endl;

  return 0;
}