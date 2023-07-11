//===--------- Start of DNNDeallocOptPass ----------===//

#include <iostream>
#include "mlir/Analysis/Liveness.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

/*!
 *  Pass that inserts memcpy to device for arguments.
 */
class DNNDeallocOptPass
    : public PassWrapper<DNNDeallocOptPass, OperationPass<ModuleOp>> {

public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    module.walk([&](DNNDeallocOp op) {
      auto deallocLoc = op.getLoc();
      auto mallocPtr = op.getDevPtr();
      auto mallocOp = mallocPtr.getDefiningOp();
      op->moveAfter(mallocOp);

      auto liveness = getAnalysis<Liveness>();
      auto liveOpVector = liveness.resolveLiveness(mallocPtr);
      auto LastDirectUsage = liveOpVector[liveOpVector.size()-1];

      auto LDUResultMallocVal = LastDirectUsage->getOperand((int)LastDirectUsage->getNumOperands()-1);
      auto LDUResultMallocOp = LDUResultMallocVal.getDefiningOp();
      if ((Operation*)mallocOp == (Operation*)LDUResultMallocOp) {
        auto LDUliveOpVector = liveness.resolveLiveness(LastDirectUsage->getResult(0));
        op->moveAfter(LDUliveOpVector[LDUliveOpVector.size()-1]);
      } else {
        op->moveAfter(LastDirectUsage);
      }

    });
  }
};

std::unique_ptr<Pass> core_dnn::createDNNDeallocOptPass() {
  return std::make_unique<DNNDeallocOptPass>();
}

