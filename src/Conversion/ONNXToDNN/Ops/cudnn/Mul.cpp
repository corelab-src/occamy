//===--------- Start of ONNXMulOpToDNN ----------===//

#include <iostream>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

struct ONNXMulOpToDNN : public ConversionPattern {
  ONNXMulOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXMulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXMulOpAdaptor operandAdaptor(operands);

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

    //---------- Making DNNMul Operation ----------//

    auto sizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);
    auto resultMalloc = rewriter.create<DNNMallocOp>(
        loc, resultMemRefType, sizeConst);

    DNNMulOp dnnMulOp;
    if(inputARank > inputBRank) {
      dnnMulOp = rewriter.create<DNNMulOp>(loc, resultMemRefType,
          inputOperandA, rewriter.getI64ArrayAttr(inputAMemRef.getShape()),
          inputOperandB, rewriter.getI64ArrayAttr(broadcastedDim),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    } else if (inputARank < inputBRank) {
      dnnMulOp = rewriter.create<DNNMulOp>(loc, resultMemRefType,
          inputOperandA, rewriter.getI64ArrayAttr(broadcastedDim),
          inputOperandB, rewriter.getI64ArrayAttr(inputBMemRef.getShape()),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    } else {
      dnnMulOp = rewriter.create<DNNMulOp>(loc, resultMemRefType,
          inputOperandA, rewriter.getI64ArrayAttr(inputAMemRef.getShape()),
          inputOperandB, rewriter.getI64ArrayAttr(inputBMemRef.getShape()),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    }

    // Insert dealloc.
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

void populateLoweringONNXMulOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXMulOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXMulOpToDNN -----------===//

