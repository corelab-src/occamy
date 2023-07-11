#include <iostream>

//===--------- Start of ONNXCastOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

enum castingMode {
  NON_CASTING,
  FTOI,
  FEXT,
  FTRUNC,
  ITOF,
  IEXT,
  ITRUNC
};

struct ONNXCastOpToDNN : public ConversionPattern {
  ONNXCastOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXCastOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXCastOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getInput();
    auto origtype = input.getType();

    auto outMemRefType = convertToMemRefType(*op->result_type_begin());
    auto elementType = outMemRefType.getElementType();

    enum castingMode castMode;

    // if same input and output type, return input
    if (origtype == elementType)
      castMode = NON_CASTING;

    if (origtype.isa<FloatType>()) {
      // cast from floating-point type to integer type
      if (elementType.isa<IntegerType>())
        castMode = FTOI;
      // cast from floating-point type to other floating-point type
      else if (elementType.isa<FloatType>()) {
        // cast from floating-point to wider floating-point
        if (origtype.getIntOrFloatBitWidth() <
            elementType.getIntOrFloatBitWidth())
          castMode = FEXT;
        // cast from floating-point to narrower floating-point
        else
          castMode = FTRUNC;
      }
    } else if (origtype.isa<IntegerType>()) {
      // cast from integer type to floating-point type
      if (elementType.isa<FloatType>())
        castMode = ITOF;
      else if (elementType.isa<IntegerType>())
        // cast from integer to wider integer
        if (origtype.getIntOrFloatBitWidth() <
            elementType.getIntOrFloatBitWidth())
          castMode = IEXT;
      // cast from integer to narrower integer
        else
          castMode = ITRUNC;
      else
        llvm_unreachable("unsupported element type");
    }


    auto outShape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outShape.size(); ++i)
      numElements *= outShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;
    //---------- Making DNNCast Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);
    auto dnnCastOp = rewriter.create<DNNCastOp>(loc,
        outMemRefType, input, outMalloc,
        rewriter.getI64ArrayAttr(outMemRefType.getShape()),
        rewriter.getI64IntegerAttr(castMode));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = dnnCastOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXCastOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXCastOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXCastOpToDNN -----------===//

