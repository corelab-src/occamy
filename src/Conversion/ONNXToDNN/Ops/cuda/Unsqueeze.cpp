#include <iostream>

//===--------- Start of ONNXUnsqueezeOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXUnsqueezeOpToDNN : public ConversionPattern {
  ONNXUnsqueezeOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXUnsqueezeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXUnsqueezeOpAdaptor operandAdaptor(operands);
    auto unsqueezeOp = dyn_cast<ONNXUnsqueezeOp>(op);

    auto input = operandAdaptor.getData();
    auto output = unsqueezeOp.getResult();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());

    auto axesValue = unsqueezeOp.getAxes();

    ArrayAttr unsqAxes;
    if(isa<mlir::ONNXConstantOp>(axesValue.getDefiningOp())) {
      auto axesConst = dyn_cast<ONNXConstantOp>(axesValue.getDefiningOp());
      if(onnx_mlir::isDenseONNXConstant(axesConst)) {
        unsqAxes = onnx_mlir::createArrayAttrFromConstantOp(rewriter, axesConst);
      }
    } else {
      assert(0 && "ONNXUnsqeezeOp: Unsupported axes value");
    }

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
        unsqAxes);

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

void populateLoweringONNXUnsqueezeOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXUnsqueezeOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXUnsqueezeOpToDNN -----------===//

