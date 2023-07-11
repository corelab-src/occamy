
//===--------- Start of ONNXConstantOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;

bool checkOpResultIsReturned(ONNXConstantOp *constantOp) {
  func::FuncOp function = onnx_mlir::getContainingFunction(constantOp->getOperation());

  bool opIsReturned = false;
  function.walk([&opIsReturned, constantOp](func::ReturnOp op) {
    auto result = constantOp->getResult();
    for (const auto &operand : op.getOperands())
      if (operand == result)
        opIsReturned = true;
  });

  return opIsReturned;
}

struct ONNXConstantOpToDNN : public ConversionPattern {

  ONNXConstantOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXConstantOp::getOperationName(), 1, context) {
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXConstantOpAdaptor operandAdaptor(operands);

    auto constantOp = llvm::dyn_cast<ONNXConstantOp>(op);
    if (constantOp.getSparseValue().has_value())
      return emitError(loc, "Only support dense values at this time");

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    MultiDialectBuilder<KrnlBuilder> create(rewriter, loc);
    auto shape = memRefType.getShape();

    Value constantGlobal = create.krnl.constant(memRefType,
        "constant_dnn_", constantOp.getValue().value());
    KrnlGlobalOp constantGlobalOp = dyn_cast<KrnlGlobalOp>(constantGlobal.getDefiningOp());

    int64_t numElements = 1;
    for (size_t i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      memRefType.getElementType().getIntOrFloatBitWidth() / 8;

    // Check if the variable is returned.
    if (checkOpResultIsReturned(&constantOp)) {
      // In this case, use an AllocOp for the constant since krnl.Global
      // operations are not mean to be returned.
      memref::AllocOp alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

      // Compute size in bytes using the input tensor.
      Value tensorSize = emitConstantOp(rewriter, loc,
          rewriter.getIntegerType(64), onnx_mlir::getMemRefEltSizeInBytes(memRefType));
      auto numElementsValue = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), numElements);
      tensorSize = rewriter.create<arith::MulIOp>(loc, tensorSize, numElementsValue);

      // Copy the value in the AllocOp.
      rewriter.create<KrnlMemcpyOp>(
          loc, alloc, constantGlobalOp.getResult(), tensorSize,
          onnx_mlir::LiteralIndexExpr(0).getValue(), onnx_mlir::LiteralIndexExpr(0).getValue());

      // Since the value is returned we need to only work with the AllocOp
      // not the KrnlGlobalOp. Globals cannot be returned.
      rewriter.replaceOp(op, alloc.getResult());
      return success();
    }

    auto int32Ty = rewriter.getIntegerType(32);
    auto int64Ty = rewriter.getIntegerType(64);

    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty, sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, memRefType, sizeConst);

    rewriter.create<DNNMemcpyOp>(loc, int32Ty,
        outMalloc.getResult(), constantGlobalOp.getResult(), sizeConst,
        rewriter.getI32IntegerAttr(1));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);

    //--------------- Lowering Pattern End ---------------//

    // Insert memcpy if this op is returned.
    Value ret;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    else
      ret = outMalloc.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXConstantOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXConstantOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXConstantOpToDNN -----------===//

