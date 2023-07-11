//===------------ Converting ONNX to DNN -----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"

using namespace mlir;

//========== Start of ConvertDNNToLLVMPass ==========//
namespace {
struct ConvertDNNToLLVMPass
    : public PassWrapper<ConvertDNNToLLVMPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-cudnn-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the DNN dialects to LLVM.";
  }
  void runOnOperation() final;
};
} // end of namespace for ConvertDNNToLLVMPass

void ConvertDNNToLLVMPass::runOnOperation() {

  ModuleOp module = getOperation();

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect, KrnlDialect,
    arith::ArithDialect, memref::MemRefDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<func::FuncOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  Value cudnnHandle;
  Value cudaStreamValue;
  LowerToLLVMOptions options(&getContext());
  generateDNNHandle(module.getContext(), module, cudnnHandle);
  generateCUDAStream(module.getContext(), module, cudaStreamValue);

  LLVMTypeConverter typeConverter(&getContext(), options);

  RewritePatternSet patterns(&getContext());

  populateDNNToLLVMConversionPatterns(
      patterns, &getContext(), typeConverter, cudnnHandle, cudaStreamValue);

  // --------------------------------------------------------- //

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> core_dnn::createConvertDNNToLLVMPass() {
  return std::make_unique<ConvertDNNToLLVMPass>();
}
//=========== End of ConvertDNNToLLVMPass ===========//

