#include <iostream>

//===--------- Start of ONNXMaxPoolSingleOutOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXMaxPoolSingleOutOpToDNN : public ConversionPattern {
  ONNXMaxPoolSingleOutOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXMaxPoolSingleOutOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXMaxPoolSingleOutOpAdaptor operandAdaptor(operands);
    auto onnxMaxPoolOp = dyn_cast<ONNXMaxPoolSingleOutOp>(op);

    auto input = operandAdaptor.getX();
    auto output = onnxMaxPoolOp.getResult();
    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());
    auto outputShape = outputMemRef.getShape();

    // Read dilations attribute if the op has.
    std::vector<int64_t> dilations;
    auto dilationsAttribute = onnxMaxPoolOp.getDilations();
    bool isDefaultDilations = true;
    if (dilationsAttribute)  {
      for (auto dilation : dilationsAttribute.value()) {
        int64_t dilationValue = dilation.cast<IntegerAttr>().getInt();
        if (dilationValue > 1 && isDefaultDilations)
          isDefaultDilations = false;
        dilations.emplace_back(dilationValue);
      }
    }
    if (isDefaultDilations)
      dilations = {};


    // Read kernel_shape attribute
    SmallVector<int64_t, 4> kernelShape;
    auto kernelShapeAttribute = onnxMaxPoolOp.getKernelShape();
    for (Attribute dim : kernelShapeAttribute.getValue())
      kernelShape.emplace_back(dim.cast<IntegerAttr>().getInt());

    if (kernelShape[0] != kernelShape[1])
      return emitError(loc, "Pooling: kernel_shape: Now only support square shaped kernels");


    // Read pads attribute
    SmallVector<int64_t, 4> paddings;
    auto padsAttribute = onnxMaxPoolOp.getPads();
    for (Attribute pad : padsAttribute.value())
      paddings.emplace_back(pad.cast<IntegerAttr>().getInt());

    if((paddings[0] != paddings[1]) || (paddings[2] != paddings[3]))
      return emitError(loc, "Pooling: Padding: Now only support symetric paddings");


    // Read strides attribute
    SmallVector<int64_t, 4> strides;
    auto stridesAttribute = onnxMaxPoolOp.getStrides();
    for (Attribute stride : stridesAttribute.value())
      strides.emplace_back(stride.cast<IntegerAttr>().getInt());

    int64_t numElements = 1;
    for (size_t i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outputMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    //-------------- Making DNNMaxPoolOperation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto resultMalloc = rewriter.create<DNNMallocOp>(loc, outputMemRef, sizeConst);
    auto dnnMaxPool = rewriter.create<DNNMaxPoolOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        resultMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        rewriter.getI64ArrayAttr(dilations),
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
      ret = dnnMaxPool.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXMaxPoolSingleOutOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXMaxPoolSingleOutOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXMaxPoolSingleOutOpToDNN -----------===//

