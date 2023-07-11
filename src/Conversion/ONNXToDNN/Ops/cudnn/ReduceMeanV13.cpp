#include <iostream>

//===--------- Start of ONNXReduceMeanV13OpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXReduceMeanV13OpToDNN : public ConversionPattern {
  ONNXReduceMeanV13OpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXReduceMeanV13Op::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXReduceMeanV13OpAdaptor operandAdaptor(operands);
    auto onnxReduceOp = dyn_cast<ONNXReduceMeanV13Op>(op);

    auto input = operandAdaptor.getData();
    auto output = onnxReduceOp.getResult();
    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());
    auto inputShape = inputMemRef.getShape();
    auto outputShape = outputMemRef.getShape();

    int64_t numElements = 1;
    for (size_t i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outputMemRef.getElementType().getIntOrFloatBitWidth() / 8;
    int32_t reduceMode = 5; /*DNN_REDUCE_TENSOR_AVG*/

    //-------------- Making DNNReduce Operation --------------//
    int64_t inRank = inputMemRef.getRank();

    // Get attributes
    auto axisAttrs = onnxReduceOp.getAxes();
    std::vector<int64_t> axes;
    if (axisAttrs) {
      for (auto axisAttr : axisAttrs.value()) {
        int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
        axis = axis >= 0 ? axis : (inRank + axis);
        assert(axis >= -inRank && axis <= inRank - 1);
        if (std::find(axes.begin(), axes.end(), axis) == axes.end())
          axes.push_back(axis);
      }
    } else {
      for (decltype(inRank) i = 0; i < inRank; ++i) {
        axes.push_back(i);
      }
    }

    // TODO: Set PreFlattenAxes for Adding op LATER!!
    std::vector<int64_t>preFlattenAxes;
    for (int64_t i=0; i<inRank; i++) {
      preFlattenAxes.push_back((int64_t)inputShape[i]);
    }
    for (auto idx: axes) {
      preFlattenAxes[idx] = 1;
    }

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto resultMalloc = rewriter.create<DNNMallocOp>(loc, outputMemRef, sizeConst);
    auto dnnReduce = rewriter.create<DNNReduceOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        resultMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        rewriter.getI64ArrayAttr(preFlattenAxes),
        rewriter.getI32IntegerAttr(reduceMode));

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = dnnReduce.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXReduceMeanV13OpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXReduceMeanV13OpToDNN>(typeConverter, context);
}
//===---------- End of ONNXReduceMeanV13OpToDNN -----------===//

