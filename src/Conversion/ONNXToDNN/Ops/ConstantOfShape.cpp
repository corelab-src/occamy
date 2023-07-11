#include <iostream>

//===--------- Start of ONNXConstantOfShapeOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXConstantOfShapeOpToDNN : public ConversionPattern {
  static int constantOfShapeID;

  ONNXConstantOfShapeOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXConstantOfShapeOp::getOperationName(), 1, context) {
      constantOfShapeID = 0;
    }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto constantofshapeOp = dyn_cast<ONNXConstantOfShapeOp>(op);

    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto shape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (size_t i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making DNNConstantOfShape Operation ----------//

    // Emit the constant global in Krnl dialect.
    auto constantGlobal = rewriter.create<KrnlGlobalOp>(loc, outMemRefType,
        /*shape=*/rewriter.getI64ArrayAttr(shape),
        /*name=*/
        rewriter.getStringAttr("constantofshape_dnn_" + std::to_string(constantOfShapeID)),
        // /*value=*/constantValue,
        /*value=*/constantofshapeOp.getValue().value(),
        /*offset=*/nullptr,
        /*alignment=*/nullptr);

    // Increment constant ID:
    constantOfShapeID++;

    Operation *parentFuncOp = op->getParentOp();
    // While parent is not a func::FuncOp and its cast to a func::FuncOp is null.
    while (!llvm::dyn_cast_or_null<func::FuncOp>(parentFuncOp))
      parentFuncOp = parentFuncOp->getParentOp();

    func::FuncOp function = cast<func::FuncOp>(parentFuncOp);
    bool opIsReturned = false;
    function.walk([&opIsReturned, &constantofshapeOp](func::ReturnOp op) {
        auto result = constantofshapeOp.getResult();
        for (const auto &operand : op.getOperands()) {
          if (operand == result)
          opIsReturned = true;
        }
      });

    // Check if the variable is returned.
    if (opIsReturned) {
      // In this case, use an AllocOp for the constant since krnl.Global
      // operations are not mean to be returned.
      memref::AllocOp alloc = rewriter.create<memref::AllocOp>(loc, outMemRefType);

      // Compute size in bytes using the input tensor.
      Value tensorSize = emitConstantOp(rewriter, loc,
          rewriter.getIntegerType(64), onnx_mlir::getMemRefEltSizeInBytes(outMemRefType));
      auto numElementsValue = emitConstantOp(
          rewriter, loc, rewriter.getIntegerType(64), numElements);
      tensorSize = rewriter.create<arith::MulIOp>(loc, tensorSize, numElementsValue);

      // Copy the value in the AllocOp.
      rewriter.create<KrnlMemcpyOp>(
          loc, alloc, constantGlobal.getResult(), tensorSize,
          onnx_mlir::LiteralIndexExpr(0).getValue(), onnx_mlir::LiteralIndexExpr(0).getValue());

      // Since the value is returned we need to only work with the AllocOp
      // not the KrnlGlobalOp. Globals cannot be returned.
      rewriter.replaceOp(op, alloc.getResult());
      return success();
    }

    auto int32Ty = rewriter.getIntegerType(32);
    auto int64Ty = rewriter.getIntegerType(64);

    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<DNNMallocOp>(loc, outMemRefType, sizeConst);

    rewriter.create<DNNMemcpyOp>(loc, int32Ty,
        outMalloc.getResult(), constantGlobal.getResult(), sizeConst,
        rewriter.getI32IntegerAttr(1));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);

    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    else
      ret = outMalloc.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

int ONNXConstantOfShapeOpToDNN::constantOfShapeID;

void populateLoweringONNXConstantOfShapeOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXConstantOfShapeOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXConstantOfShapeOpToDNN -----------===//

