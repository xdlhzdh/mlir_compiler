#include "ConvBNFusionPattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;

LogicalResult
ConvBNFusionPattern::matchAndRewrite(stablehlo::BatchNormInferenceOp bnOp,
                                     PatternRewriter &rewriter) const {

  Value convOut = bnOp.getOperand();
  auto convOp = convOut.getDefiningOp<stablehlo::ConvolutionOp>();
  if (!convOp)
    return failure();

  Value gamma = bnOp.getScale();
  Value beta = bnOp.getOffset();
  Value mean = bnOp.getMean();
  Value var = bnOp.getVariance();

  auto gammaCst = gamma.getDefiningOp<arith::ConstantOp>();
  auto betaCst = beta.getDefiningOp<arith::ConstantOp>();
  auto meanCst = mean.getDefiningOp<arith::ConstantOp>();
  auto varCst = var.getDefiningOp<arith::ConstantOp>();

  if (!gammaCst || !betaCst || !meanCst || !varCst)
    return failure();

  auto gammaAttr = cast<DenseFPElementsAttr>(gammaCst.getValue());
  auto betaAttr = cast<DenseFPElementsAttr>(betaCst.getValue());
  auto meanAttr = cast<DenseFPElementsAttr>(meanCst.getValue());
  auto varAttr = cast<DenseFPElementsAttr>(varCst.getValue());

  float epsilon = bnOp.getEpsilon().convertToFloat();

  SmallVector<float> A;
  SmallVector<float> B;
  auto gammaVals = gammaAttr.template getValues<float>();
  auto betaVals = betaAttr.template getValues<float>();
  auto meanVals = meanAttr.template getValues<float>();
  auto varVals = varAttr.template getValues<float>();

  for (size_t i = 0; i < gammaAttr.size(); ++i) {
    float a = gammaVals[i] / std::sqrt(varVals[i] + epsilon);
    float b = betaVals[i] - meanVals[i] * a;
    A.push_back(a);
    B.push_back(b);
  }

  Value weight = convOp.getRhs();
  auto weightCst = weight.getDefiningOp<arith::ConstantOp>();
  if (!weightCst)
    return failure();

  auto weightAttr = cast<DenseFPElementsAttr>(weightCst.getValue());
  SmallVector<float> newWeightVals;
  auto weightVals = weightAttr.template getValues<float>();
  int OC = A.size();
  int idx = 0;
  for (float w : weightVals) {
    int oc = idx % OC;
    newWeightVals.push_back(w * A[oc]);
    idx++;
  }

  auto newWeightAttr =
      DenseFPElementsAttr::get(weightAttr.getType(), ArrayRef(newWeightVals));
  Value newWeight =
      arith::ConstantOp::create(rewriter, bnOp.getLoc(), newWeightAttr);

  OperationState state(bnOp.getLoc(),
                       stablehlo::ConvolutionOp::getOperationName());
  stablehlo::ConvolutionOp::build(rewriter, state, convOp.getResult().getType(),
                                  ValueRange{convOp.getLhs(), newWeight},
                                  convOp->getAttrs());
  auto *newConvOp = rewriter.create(state);
  rewriter.replaceOp(bnOp,
                     cast<stablehlo::ConvolutionOp>(newConvOp).getResult());
  return success();
}
