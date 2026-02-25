#include "pass/remove_trivial_block_passes.h"
#include "pass/remove_trivial_loop_passes.h"
#include "pass/remove_zero_trip_loop_passes.h"
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
                  if (Name == "my-remove-trivial-block") {
                    FPM.addPass(RemoveTrivialBlockPass());
                    return true;
                  }
                  if (Name == "my-remove-trivial-loop") {
                    FPM.addPass(RemoveTrivialLoopPass());
                    return true;
                  }
                  if (Name == "my-remove-zero-trip-loop") {
                    FPM.addPass(RemoveZeroTripLoopPass());
                    return true;
                  }
                  return false;
                });
          }};
}
