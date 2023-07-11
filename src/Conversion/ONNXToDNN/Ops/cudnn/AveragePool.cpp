#include <iostream>

//===--------- Start of ONNXAveragePoolOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXAveragePoolOpToDNN : public ConversionPattern {
  ONNXAveragePoolOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXAveragePoolOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXAveragePoolOpAdaptor operandAdaptor(operands);
    auto onnxAveragePoolOp = dyn_cast<ONNXAveragePoolOp>(op);

    auto input = operandAdaptor.getX();
    auto output = onnxAveragePoolOp.getResult();
    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());
    auto outputShape = outputMemRef.getShape();


    // Read kernel_shape attribute
    SmallVector<int64_t, 4> kernelShape;
    auto kernelShapeAttribute = onnxAveragePoolOp.getKernelShape();
    for (Attribute dim : kernelShapeAttribute.getValue())
      kernelShape.emplace_back(dim.cast<IntegerAttr>().getInt());

    if (kernelShape[0] != kernelShape[1])
      return emitError(loc, "Pooling: kernel_shape: Now only support square shaped kernels");


    // Read pads attribute
    SmallVector<int64_t, 4> paddings;
    if (auto padsAttribute = onnxAveragePoolOp.getPads()) {
      for (Attribute pad : padsAttribute.value())
        paddings.emplace_back(pad.cast<IntegerAttr>().getInt());

      if((paddings[0] != paddings[1]) || (paddings[2] != paddings[3]))
        return emitError(loc, "Pooling: Padding: Now only support symetric paddings");
    } else {
      paddings.push_back(0);
      paddings.push_back(0);
      paddings.push_back(0);
      paddings.push_back(0);
    }


    // Read strides attribute
    SmallVector<int64_t, 4> strides;
    auto stridesAttribute = onnxAveragePoolOp.getStrides();
    for (Attribute stride : stridesAttribute.value())
      strides.emplace_back(stride.cast<IntegerAttr>().getInt());

    int64_t numElements = 1;
    for (size_t i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outputMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    //-------------- Making DNNAveragePoolOperation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto resultMalloc = rewriter.create<DNNMallocOp>(loc, outputMemRef, sizeConst);
    auto dnnAveragePool = rewriter.create<DNNAveragePoolOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        resultMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        rewriter.getI64ArrayAttr(kernelShape),
        rewriter.getI64ArrayAttr(paddings),
        rewriter.getI64ArrayAttr(strides));

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = dnnAveragePool.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXAveragePoolOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXAveragePoolOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXAveragePoolOpToDNN -----------===//

