#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

using namespace llvm;

namespace simple_llvm_ir {
// Goal: Create an LLVM IR function equivalent to C++: int add(int a, int b) {
// return a + b; }
inline void createAddFunction(Module &TheModule, LLVMContext &TheContext) {
  // 1. Setup IR builder
  IRBuilder<> Builder(TheContext);

  // 2. Define function type: returns i32, parameters are (i32, i32)
  Type *I32Type = Type::getInt32Ty(TheContext);
  std::vector<Type *> Ints(2, I32Type); // Two i32 parameters
  FunctionType *FuncType = FunctionType::get(
      I32Type, Ints,
      false); // Return type, parameter list, is varargeter list, is vararg

  // 3. Create function declaration
  Function *AddFunc =
      Function::Create(FuncType,                  // Function type
                       Function::ExternalLinkage, // Linkage type
                       "add",                     // Function name
                       TheModule                  // Module
      );

  // Set parameter names
  unsigned Idx = 0;
  for (auto &Arg : AddFunc->args()) {
    Arg.setName(Idx == 0 ? "a" : "b");
    Idx++;
  }

  // Get named parameters
  Function::arg_iterator Args = AddFunc->arg_begin();
  Value *A = &*Args;
  Args++;            // Parameter 'a'
  Value *B = &*Args; // Parameter 'b'ter 'b'

  // 4. Create basic block
  BasicBlock *EntryBB = BasicBlock::Create(TheContext, "entry", AddFunc);

  // 5. Position builder at the end of basic block
  Builder.SetInsertPoint(EntryBB);

  // 6. Insert instructions
  // Create add instruction: result = a + b
  Value *Result = Builder.CreateAdd(A, B, "result");

  // Create return instruction: return result
  Builder.CreateRet(Result);
}

} // namespace simple_llvm_ir
