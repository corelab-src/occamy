//===--------- Start of ONNXSqrtOpToDNN ----------===//

#include <iostream>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

struct ONNXSqrtOpToDNN : public ConversionPattern {
  ONNXSqrtOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSqrtOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSqrtOpAdaptor operandAdaptor(operands);
    auto input = operandAdaptor.getX();
    auto resultMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t resultSize = 1, resultNumElement = 1;

    for (unsigned int i=0; i<resultMemRefType.getShape().size(); i++) {
      resultNumElement *= resultMemRefType.getShape()[i];
    }
    resultSize = resultNumElement *
      resultMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    auto I64Ty = rewriter.getIntegerType(64);

    //---------- Making DNNSqrt Operation ----------//

    auto sizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);
    auto resultMalloc = rewriter.create<DNNMallocOp>(
        loc, resultMemRefType, sizeConst);
    auto dnnSqrt = rewriter.create<DNNSqrtOp>( loc, resultMemRefType,
        input, resultMalloc,
        rewriter.getI64ArrayAttr(resultMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = dnnSqrt.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSqrtOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSqrtOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXSqrtOpToDNN -----------===//
