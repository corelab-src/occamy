//===--------- Start of ONNXConstantHoistingPass ----------===//

#include <iostream>
#include "mlir/Analysis/Liveness.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

/*!
 *  Pass that inserts memcpy to device for arguments.
 */

class ONNXConstantHoistingPass
    : public PassWrapper<ONNXConstantHoistingPass, OperationPass<func::FuncOp>> {

public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto &parentBlock = funcOp.front();
    OpBuilder builder(&funcOp.front().front());
    auto locOp = &parentBlock.front();

    SmallVector<Operation*, 1> constantOps;

    funcOp.walk([&](ONNXConstantOp op) {
      constantOps.emplace_back(op.getOperation());
    });

    for (long unsigned int i=0; i<constantOps.size(); i++) {
      constantOps[i]->moveAfter(locOp);
      locOp = constantOps[i];
    }
  }
};

std::unique_ptr<mlir::Pass> core_dnn::createONNXConstantHoistingPass() {
  return std::make_unique<ONNXConstantHoistingPass>();
}

