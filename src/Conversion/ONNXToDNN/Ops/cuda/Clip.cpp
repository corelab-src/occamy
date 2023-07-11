#include <iostream>

//===--------- Start of ONNXClipOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXClipOpToDNN : public ConversionPattern {
  ONNXClipOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXClipOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXClipOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getInput();
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto outShape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outShape.size(); ++i)
      numElements *= outShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    float minVal = -1;
    float maxVal = -1;
    auto minMalloc = operandAdaptor.getMin().getDefiningOp();
    for (Operation* user: minMalloc->getUsers()) {
      if (auto memcpyOp = dyn_cast<DNNMemcpyOp>(user)) {
        auto globalOp = dyn_cast<KrnlGlobalOp>(memcpyOp.getSrc().getDefiningOp());
        auto valueAttr = globalOp.getValueAttr().cast<DenseElementsAttr>();
        minVal = (float)valueAttr.getValues<FloatAttr>()[0].cast<FloatAttr>().getValueAsDouble();
      }
    }
    auto maxMalloc = operandAdaptor.getMax().getDefiningOp();
    for (Operation* user: maxMalloc->getUsers()) {
      if (auto memcpyOp = dyn_cast<DNNMemcpyOp>(user)) {
        auto globalOp = dyn_cast<KrnlGlobalOp>(memcpyOp.getSrc().getDefiningOp());
        auto valueAttr = globalOp.getValueAttr().cast<DenseElementsAttr>();
        maxVal = (float)valueAttr.getValues<FloatAttr>()[0].cast<FloatAttr>().getValueAsDouble();
      }
    }

    //---------- Making DNNClip Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
      sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);
    auto dnnClipOp = rewriter.create<DNNClipOp>(loc, outMemRefType,
        input, FloatAttr::get(rewriter.getF32Type(), minVal),
        FloatAttr::get(rewriter.getF32Type(), maxVal),
        outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = dnnClipOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXClipOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXClipOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXClipOpToDNN -----------===//

