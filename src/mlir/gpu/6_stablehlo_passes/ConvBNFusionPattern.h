#ifndef CONV_BN_FUSION_PATTERN_H
#define CONV_BN_FUSION_PATTERN_H

// P5 (6_stablehlo_passes): GreedyPatternRewrite for stablehlo.convolution + batch_norm_inference fusion.

#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {

class ConvBNFusionPattern
    : public OpRewritePattern<stablehlo::BatchNormInferenceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::BatchNormInferenceOp bnOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir

#endif
