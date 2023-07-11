#include <iostream>

//===--------- Start of ONNXReshapeOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXReshapeOpToDNN : public ConversionPattern {
  ONNXReshapeOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXReshapeOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getData();
    auto shape = operandAdaptor.getShape();

    auto inputMemRefType = convertToMemRefType(input.getType());
    auto shapeMemRefType = convertToMemRefType(shape.getType());
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto outputShape = outMemRefType.getShape();
    auto shapeShape = shapeMemRefType.getShape();

    size_t shapeDim = shapeShape.size();
    assert((shapeDim == 1) && "Only support 1D shape operand");

    int64_t numElements = 1;
    for (unsigned int i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making DNNReshape Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);
    auto dnnReshapeOp = rewriter.create<DNNReshapeOp>(loc, outMemRefType,
        input, rewriter.getI64ArrayAttr(inputMemRefType.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = dnnReshapeOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXReshapeOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXReshapeOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXReshapeOpToDNN -----------===//

