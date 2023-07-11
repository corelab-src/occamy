#include <iostream>

//===--------- Start of ONNXGemmOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXGemmOpToDNN : public ConversionPattern {
  ONNXGemmOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXGemmOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXGemmOpAdaptor operandAdaptor(operands);
    auto onnxGemmOp = dyn_cast<ONNXGemmOp>(op);

    auto inputA = operandAdaptor.getA();
    auto inputB = operandAdaptor.getB();
    auto inputC = operandAdaptor.getC();
    auto output = onnxGemmOp.getResult();

    auto alpha = onnxGemmOp.getAlpha().convertToFloat();
    auto beta = onnxGemmOp.getBeta().convertToFloat();
    auto transA = onnxGemmOp.getTransA();
    auto transB = onnxGemmOp.getTransB();

    auto inputAMemRef = convertToMemRefType(inputA.getType());
    auto inputBMemRef = convertToMemRefType(inputB.getType());
    auto inputCMemRef = convertToMemRefType(inputC.getType());
    auto outputYMemRef = convertToMemRefType(output.getType());

    auto inputAShape = inputAMemRef.getShape();
    auto inputBShape = inputBMemRef.getShape();
    auto outputYShape = outputYMemRef.getShape();

    // Compute broadcasting Dim for InputC
    SmallVector<int64_t> broadcastedDim;
    broadcastOnlyDimension(&broadcastedDim, outputYMemRef, inputCMemRef);

    int64_t numElements = 1;
    for (size_t i = 0; i < outputYShape.size(); i++)
      numElements *= outputYShape[i];
    int64_t sizeBytes = numElements *
      outputYMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    //-------------- Making DNNGemmOperation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto matmulMalloc = rewriter.create<DNNMallocOp>(loc, outputYMemRef, sizeConst);

    // create dnnMatmul2d operator
    rewriter.create<DNNMatmul2dOp>(loc, outputYMemRef,
        inputA, rewriter.getI64ArrayAttr(inputAShape),
        inputB, rewriter.getI64ArrayAttr(inputBShape),
        matmulMalloc, rewriter.getI64ArrayAttr(outputYShape),
        rewriter.getF32FloatAttr(alpha),
        rewriter.getF32FloatAttr(beta),
        rewriter.getI64IntegerAttr(transA),
        rewriter.getI64IntegerAttr(transB));

    // create dnnadd operator
    auto addOp = rewriter.create<DNNAddOp>(loc, outputYMemRef,
        matmulMalloc, rewriter.getI64ArrayAttr(outputYShape),
        // create dnnadd operator
        inputC, rewriter.getI64ArrayAttr(broadcastedDim),
        FloatAttr::get(rewriter.getF32Type(), 1.f),
        matmulMalloc, rewriter.getI64ArrayAttr(outputYShape));

    // Insert dealloc.
    insertDealloc(matmulMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, matmulMalloc, loc, rewriter);
    if (!ret)
      ret = addOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXGemmOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXGemmOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXGemmOpToDNN -----------===//

