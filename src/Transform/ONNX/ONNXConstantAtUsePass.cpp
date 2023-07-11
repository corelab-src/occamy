//===--------- Start of ONNXConstantAtUsePass ----------===//

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

class ONNXConstantAtUsePass
    : public PassWrapper<ONNXConstantAtUsePass, OperationPass<func::FuncOp>> {

public:
  struct constMoves {
    Operation* tgtConst;
    Operation* tgtUser;
  };

  SmallVector<constMoves*, 1> moveVec;

  void findFirstUserAndStore(Operation* targetOp) {
    for (Operation* opIt = targetOp;
        opIt != (&getOperation().back().back());
        opIt = opIt->getNextNode()) {
      for (auto operand: opIt->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (defOp == targetOp) {
            constMoves* moves = new constMoves{targetOp, opIt};
            moveVec.emplace_back(moves);
            return;
          }
        }
      }
    }
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(&funcOp.front().front());

    funcOp.walk([&](ONNXConstantOp constOp) {
        Operation* op = constOp.getOperation();
        findFirstUserAndStore(op);
    });

    for (long unsigned int i=0; i<moveVec.size(); i++) {
      constMoves* elem = moveVec[i];
      elem->tgtConst->moveBefore(elem->tgtUser);
    }
  }
};

std::unique_ptr<mlir::Pass> core_dnn::createONNXConstantAtUsePass() {
  return std::make_unique<ONNXConstantAtUsePass>();
}

