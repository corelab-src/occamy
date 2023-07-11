#include <iostream>

//===--------- Start of ONNXUnsqueezeV11OpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXUnsqueezeV11OpToDNN : public ConversionPattern {
  ONNXUnsqueezeV11OpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXUnsqueezeV11Op::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXUnsqueezeV11OpAdaptor operandAdaptor(operands);
    auto unsqueezeV11Op = dyn_cast<ONNXUnsqueezeV11Op>(op);

    auto input = operandAdaptor.getData();
    auto output = unsqueezeV11Op.getResult();
    auto axes = unsqueezeV11Op.getAxes();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());
    auto outputShape = outputMemRef.getShape();

    auto outMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;
    //-------------- Making DNNUnsqueeze Operation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);
    auto dnnUnsqueeze = rewriter.create<DNNUnsqueezeOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        axes);

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = dnnUnsqueeze.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXUnsqueezeV11OpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXUnsqueezeV11OpToDNN>(typeConverter, context);
}
//===---------- End of ONNXUnsqueezeV11OpToDNN -----------===//

