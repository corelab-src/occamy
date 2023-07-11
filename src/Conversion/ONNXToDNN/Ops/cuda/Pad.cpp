#include <iostream>

//===--------- Start of ONNXPadOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXPadOpToDNN : public ConversionPattern {
  ONNXPadOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXPadOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXPadOpAdaptor operandAdaptor(operands);
    auto padOp = dyn_cast<ONNXPadOp>(op);

    auto input = operandAdaptor.getData();
    auto output = padOp.getResult();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());

    auto padValue = padOp.getPads();

    // ArrayAttr padVals;
    int64_t pad = -20;
    if(isa<mlir::ONNXConstantOp>(padValue.getDefiningOp())) {
      auto padAmount = dyn_cast<ONNXConstantOp>(padValue.getDefiningOp());
      if(onnx_mlir::isDenseONNXConstant(padAmount)) {
        auto padDense = padAmount.getValueAttr().dyn_cast<DenseElementsAttr>();
        for (auto val: padDense.getValues<IntegerAttr>()) {
          pad = val.getInt();
          break;
        }
      }
    } else {
      assert(0 && "ONNXPadOp: Unsupported constant value");
    }

    if(pad == 0) {
      rewriter.replaceOp(op, input);
      auto padsArgOp = padOp.getPads().getDefiningOp();
      auto constantArgOp = padOp.getConstantValue().getDefiningOp();
      // rewriter.eraseOp(padsArgOp);
      // rewriter.eraseOp(constantArgOp);
    } else {
      assert(0 && "ONNXPadOp: needs to support extra pad");
    }

    return success();
  }
};

void populateLoweringONNXPadOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXPadOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXPadOpToDNN -----------===//

