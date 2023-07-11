//===------------ Converting ONNX to DNN -----------===//

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"

using namespace mlir;
using namespace onnx_mlir;

//========== Start of ConvertONNXToDNNPass ==========//
namespace {
struct ConvertONNXToDNNPass
    : public PassWrapper<ConvertONNXToDNNPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-onnx-to-dnn"; }

  StringRef getDescription() const override {
    return "Lower the ONNX dialects to DNN.";
  }
  void runOnOperation() final;
};
} // end of namespace for ConvertONNXToDNNPass

void ConvertONNXToDNNPass::runOnOperation() {

  ModuleOp module = getOperation();
  ConversionTarget target(getContext());

  target.addLegalDialect<DNNDialect, KrnlDialect, arith::ArithDialect, memref::MemRefDialect>();

  RewritePatternSet patterns(&getContext());

  // Convert TensorType to MemRef
  onnx_mlir::KrnlTypeConverter krnlTypeConverter;
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // func::FuncOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isSignatureLegal(op.getFunctionType());
  });

  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return krnlTypeConverter.isLegal(op);
  });

  // Operations that are legal only if types are not tensors.
  target.addDynamicallyLegalOp<func::ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
      [](Type type) { return type.isa<TensorType>();  });
  });

  // Type conversion for function signatures.
  // Call MLIR func::FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, krnlTypeConverter);
  populateCallOpTypeConversionPattern(patterns, krnlTypeConverter);
  populateReturnOpTypeConversionPattern(patterns, krnlTypeConverter);

  // ----------- Adding Patterns for Lowering Pass ----------- //

  // ===------------------ Constants ---------------------===//
  populateLoweringONNXConstantOfShapeOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXConstantOpToDNNPattern(patterns, krnlTypeConverter, &getContext());

  // ===----------------- DNN -------------------=== //
  populateLoweringONNXConvOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReluOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSigmoidOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXTanhOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXAddOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSubOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXMulOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSqrtOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReduceMeanOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReduceMeanV13OpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXMaxPoolSingleOutOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXAveragePoolOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSoftmaxOpToDNNPattern(patterns, krnlTypeConverter, &getContext());


  // ===---------------- cuBLAS -------------------=== //
  populateLoweringONNXGemmOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXMatMulOpToDNNPattern(patterns, krnlTypeConverter, &getContext());

  // ===----------------- CUDA --------------------=== //
  populateLoweringONNXCastOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXClipOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXGatherOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXExpandOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXConcatOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXReshapeOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXNonZeroOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXFlattenOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXUnsqueezeOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXUnsqueezeV11OpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSqueezeOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXSqueezeV11OpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXTransposeOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXDivOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXPowOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXErfOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXPReluOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXPadOpToDNNPattern(patterns, krnlTypeConverter, &getContext());
  populateLoweringONNXLeakyReluOpToDNNPattern(patterns, krnlTypeConverter, &getContext());

  // --------------------------------------------------------- //

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> core_dnn::createConvertONNXToDNNPass() {
  return std::make_unique<ConvertONNXToDNNPass>();
}
//=========== End of ConvertONNXToDNNPass ===========//

