#include "pass/simple_pass_passes.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SimplePass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "my-peephole") {
                    FPM.addPass(SimplePeepholePass());
                    return true;
                  }
                  if (Name == "my-cfg") {
                    FPM.addPass(RemoveEmptyBlockPass());
                    return true;
                  }
                  return false;
                });
          }};
}
