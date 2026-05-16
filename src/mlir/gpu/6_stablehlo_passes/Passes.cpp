// P5 (6_stablehlo_passes): MLIR plugin — StableHLO dialect Pass (e.g. Conv+BN fusion) for mlir-opt.
#include "ConvBNFusionPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

namespace {

struct ConvBNFusionPass
    : public PassWrapper<ConvBNFusionPass, OperationPass<func::FuncOp>> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvBNFusionPattern>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), frozen))) {
      signalPassFailure();
    }
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION, "MyPass", "v0.1", []() {
        mlir::registerPassPipeline(
            "conv-bn-fusion", "Fuse Conv and BatchNorm (StableHLO)",
            [](mlir::OpPassManager &pm, llvm::StringRef options,
               mlir::function_ref<mlir::LogicalResult(const llvm::Twine &)>
                   errorHandler) {
              if (!options.empty())
                return mlir::failure();
              pm.addNestedPass<mlir::func::FuncOp>(
                  std::make_unique<ConvBNFusionPass>());
              return mlir::success();
            },
            [](mlir::function_ref<void(const mlir::detail::PassOptions &)>) {});
      }};
}

// 同一 .so 作为 dialect 插件：只注册 StablehloDialect，避免拉入 Chlo/Vhlo
// 等依赖
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "StablehloDialect", "v0.1",
          [](mlir::DialectRegistry *registry) {
            if (registry)
              registry->insert<mlir::stablehlo::StablehloDialect>();
          }};
}
