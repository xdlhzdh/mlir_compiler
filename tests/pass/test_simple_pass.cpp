#include "pass/simple_pass_passes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/SourceMgr.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llvm;
using namespace llvm::PatternMatch;

static std::unique_ptr<Module> parseIR(LLVMContext& Ctx, const char* IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, Ctx);
}

TEST(SimplePass_Src, PeepholeAddZero) {
  LLVMContext Ctx;
  const char* IR = R"(
    define i32 @f(i32 %x) {
      %a = add i32 %x, 0
      ret i32 %a
    }
  )";
  auto M = parseIR(Ctx, IR);
  ASSERT_TRUE(M) << "parse IR failed";
  Function* F = M->getFunction("f");
  ASSERT_TRUE(F) << "no function f";

  FunctionAnalysisManager FAM;
  SimplePeepholePass().run(*F, FAM);

  // After pass: no "add i32 %x, 0", ret should use %x directly
  bool hasAddZero = false;
  for (auto& BB : *F) {
    for (auto& I : BB) {
      if (auto* Add = dyn_cast<BinaryOperator>(&I)) {
        if (Add->getOpcode() == Instruction::Add &&
            match(Add, m_Add(m_Value(), m_Zero())))
          hasAddZero = true;
      }
    }
  }
  EXPECT_FALSE(hasAddZero) << "peephole should remove add x, 0";

  // Return value should be the argument
  Instruction* Ret = F->getEntryBlock().getTerminator();
  ASSERT_TRUE(isa<ReturnInst>(Ret));
  EXPECT_EQ(Ret->getOperand(0), F->getArg(0)) << "ret should use %x";
}

TEST(SimplePass_Src, RemoveEmptyBlock) {
  LLVMContext Ctx;
  const char* IR = R"(
    define i32 @g(i32 %x) {
    entry:
      br label %mid
    mid:
      br label %exit
    exit:
      ret i32 %x
    }
  )";
  auto M = parseIR(Ctx, IR);
  ASSERT_TRUE(M) << "parse IR failed";
  Function* F = M->getFunction("g");
  ASSERT_TRUE(F) << "no function g";

  FunctionAnalysisManager FAM;
  RemoveEmptyBlockPass().run(*F, FAM);

  // After pass: empty jump blocks removed, entry -> exit
  unsigned NumBlocks = 0;
  for (auto& BB : *F)
    ++NumBlocks;
  EXPECT_LE(NumBlocks, 2u) << "empty blocks should be removed";
}
