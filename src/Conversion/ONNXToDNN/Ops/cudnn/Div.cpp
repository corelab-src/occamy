//===--------- Start of ONNXDivOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

struct ONNXDivOpToDNN : public ConversionPattern {
  ONNXDivOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXDivOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXDivOpAdaptor operandAdaptor(operands);

    auto inputOperandA = operandAdaptor.getA();
    auto inputOperandB = operandAdaptor.getB();
    auto inputAMemRef = convertToMemRefType(inputOperandA.getType());
    auto inputBMemRef = convertToMemRefType(inputOperandB.getType());

    int inputARank = inputAMemRef.getShape().size();
    int inputBRank = inputBMemRef.getShape().size();

    auto resultMemRefType = convertToMemRefType(*op->result_type_begin());

    // Compute broadcasting Dim for Input
    SmallVector<int64_t> broadcastedDim;
    broadcastOnlyDimension(&broadcastedDim, inputAMemRef, inputBMemRef);

    int64_t resultSize = 1, resultNumElement = 1;

    for (unsigned int i=0; i<resultMemRefType.getShape().size(); i++) {
      resultNumElement *= resultMemRefType.getShape()[i];
    }
    resultSize = resultNumElement *
      resultMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    auto I64Ty = rewriter.getIntegerType(64);

    //---------- Making DNNDiv Operation ----------//

    auto sizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);
    auto reciprocalResultMalloc = rewriter.create<DNNMallocOp>(
        loc, resultMemRefType, sizeConst);
    auto resultMalloc = rewriter.create<DNNMallocOp>(
        loc, resultMemRefType, sizeConst);

    ArrayAttr AarrayAttr;
    ArrayAttr BarrayAttr;
    if(inputARank > inputBRank) {
      AarrayAttr = rewriter.getI64ArrayAttr(inputAMemRef.getShape());
      BarrayAttr = rewriter.getI64ArrayAttr(broadcastedDim);
    } else if (inputARank < inputBRank) {
      AarrayAttr = rewriter.getI64ArrayAttr(broadcastedDim);
      BarrayAttr = rewriter.getI64ArrayAttr(inputBMemRef.getShape());
    } else {
      AarrayAttr = rewriter.getI64ArrayAttr(inputAMemRef.getShape());
      BarrayAttr = rewriter.getI64ArrayAttr(inputBMemRef.getShape());
    }

    // create dnnReciprocal operator
    rewriter.create<DNNReciprocalOp>(
        loc, resultMemRefType, inputOperandB,
        reciprocalResultMalloc, BarrayAttr);

    auto dnnMulOp = rewriter.create<DNNMulOp>(loc, resultMemRefType,
          inputOperandA, AarrayAttr,
          reciprocalResultMalloc, BarrayAttr,
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(reciprocalResultMalloc, loc, rewriter);
    insertDealloc(resultMalloc, loc, rewriter);

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = dnnMulOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXDivOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXDivOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXDivOpToDNN -----------===//

