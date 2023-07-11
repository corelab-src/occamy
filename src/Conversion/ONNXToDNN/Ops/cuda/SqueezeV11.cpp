#include <iostream>

//===--------- Start of ONNXSqueezeV11OpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXSqueezeV11OpToDNN : public ConversionPattern {
  ONNXSqueezeV11OpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSqueezeV11Op::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSqueezeV11OpAdaptor operandAdaptor(operands);
    auto squeezeV11Op = dyn_cast<ONNXSqueezeV11Op>(op);

    auto input = operandAdaptor.getData();
    auto output = squeezeV11Op.getResult();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());

    // Assume that `axes` has been validated by shape inference.
    // So, here we just get it.
    ArrayAttr axisAttrs = llvm::dyn_cast<ONNXSqueezeV11Op>(op).getAxesAttr();
    SmallVector<int64_t, 4> axesVector;
    bool isAxesGiven = false;
    for (auto axisAttr : axisAttrs.getValue()) {
      int axis = axisAttr.cast<IntegerAttr>().getInt();
      isAxesGiven = true;
      axesVector.emplace_back(axis);
    }

    if(!isAxesGiven) {
      for (int i = 0; i < 4; i++) {
        axesVector.emplace_back(UINT64_MAX);
      }
    }

    auto outputShape = outputMemRef.getShape();
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;
    //-------------- Making DNNSqueeze Operation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);
    auto dnnSqueeze = rewriter.create<DNNSqueezeOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        rewriter.getI64ArrayAttr(axesVector));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = dnnSqueeze.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSqueezeV11OpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSqueezeV11OpToDNN>(typeConverter, context);
}
//===---------- End of ONNXSqueezeV11OpToDNN -----------===//

