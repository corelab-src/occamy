//===--------- Start of ONNXAddOpToDNN ----------===//

#include <iostream>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

struct ONNXAddOpToDNN : public ConversionPattern {
  ONNXAddOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXAddOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXAddOpAdaptor operandAdaptor(operands);

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

    unsigned int i;
    int64_t resultSize = 1, resultNumElement = 1;

    for (i=0;i<resultMemRefType.getShape().size();i++) {
      resultNumElement *= resultMemRefType.getShape()[i];
    }
    resultSize = resultNumElement *
      resultMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    auto I64Ty = rewriter.getIntegerType(64);

    //---------- Making DNNAdd Operation ----------//

    auto sizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);
    auto resultMalloc = rewriter.create<DNNMallocOp>(
        loc, resultMemRefType, sizeConst);

    DNNAddOp dnnAddOp;
    if(inputARank > inputBRank) {
      dnnAddOp = rewriter.create<DNNAddOp>(loc, resultMemRefType,
          inputOperandA, rewriter.getI64ArrayAttr(inputAMemRef.getShape()),
          inputOperandB, rewriter.getI64ArrayAttr(broadcastedDim),
          FloatAttr::get(rewriter.getF32Type(), 1.f),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    } else if (inputARank < inputBRank) {
      dnnAddOp = rewriter.create<DNNAddOp>(loc, resultMemRefType,
          inputOperandB, rewriter.getI64ArrayAttr(inputBMemRef.getShape()),
          inputOperandA, rewriter.getI64ArrayAttr(broadcastedDim),
          FloatAttr::get(rewriter.getF32Type(), 1.f),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    } else {
      dnnAddOp = rewriter.create<DNNAddOp>(loc, resultMemRefType,
          inputOperandA, rewriter.getI64ArrayAttr(inputAMemRef.getShape()),
          inputOperandB, rewriter.getI64ArrayAttr(inputBMemRef.getShape()),
          FloatAttr::get(rewriter.getF32Type(), 1.f),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    }

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = dnnAddOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXAddOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXAddOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXAddOpToDNN -----------===//
