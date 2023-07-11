#include <iostream>

//===--------- Start of ONNXSoftmaxOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXSoftmaxOpToDNN : public ConversionPattern {
  ONNXSoftmaxOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSoftmaxOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSoftmaxOpAdaptor operandAdaptor(operands);
    auto softmaxOp = dyn_cast<ONNXSoftmaxOp>(op);

    auto input = operandAdaptor.getInput();
    auto axis = softmaxOp.getAxis();
    auto output = softmaxOp.getResult();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());

    auto outputShape = outputMemRef.getShape();

    int64_t numElements = 1;
    for (unsigned int i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outputMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    //-------------- Making DNNSoftmaxOperation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto resultMalloc = rewriter.create<DNNMallocOp>(loc, outputMemRef, sizeConst);
    auto dnnsoftmax = rewriter.create<DNNSoftmaxOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        resultMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        rewriter.getI64IntegerAttr(axis));

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = dnnsoftmax.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSoftmaxOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSoftmaxOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXSoftmaxOpToDNN -----------===//

