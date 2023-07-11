#include <iostream>

//===--------- Start of ONNXFlattenOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXFlattenOpToDNN : public ConversionPattern {
  ONNXFlattenOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXFlattenOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXFlattenOpAdaptor operandAdaptor(operands);
    auto flattenOp = dyn_cast<ONNXFlattenOp>(op);

    auto input = operandAdaptor.getInput();
    auto output = flattenOp.getResult();
    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());
    auto inputShape = inputMemRef.getShape();
    auto inputRank = inputShape.size();
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto axisValue = flattenOp.getAxis();
    if (axisValue < 0)
      axisValue = inputRank + axisValue;

    auto outShape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outShape.size(); ++i)
      numElements *= outShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;
    //-------------- Making DNNFlatten Operation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
      sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);
    auto dnnFlatten = rewriter.create<DNNFlattenOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        rewriter.getI32IntegerAttr(axisValue));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = dnnFlatten.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXFlattenOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXFlattenOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXFlattenOpToDNN -----------===//

