//===--------- Start of mallocPoolOptPass ----------===//
#include <iostream>
#include <map>
#include "mlir/Analysis/Liveness.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Transform/DNN/mallocPoolOpt.hpp"

/*!
 *  Pass that make memory pool on GPU.
 */
class mallocPoolOptPass
    : public PassWrapper<mallocPoolOptPass, OperationPass<func::FuncOp>> {

public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getOperation());

    SmallVector<Operation*, 1> replaceVec;
    SmallVector<Operation*, 1> removeDeallocVec;
    SmallVector<Operation*, 1> removeDIVVec;

    int64_t maxAllocByte = 0;

    auto loc = funcOp.getLoc();
    Operation* memPoolInitOp = nullptr;
    auto &parentBlock = funcOp.front();
    memPoolInitOp = generateMemPoolInit (builder, maxAllocByte, &parentBlock.front());
    auto memPoolSize = dyn_cast<arith::ConstantOp>(memPoolInitOp->getOperand(0).getDefiningOp());
    Operation* firstAllocOp = nullptr;

    // Run memory allocation simulation for setting an offset of the malloc op
    for(Operation* op = (&funcOp.front().front()); op != (&funcOp.back().back()); op = op->getNextNode()){
      if (isa<mlir::DNNMallocOp>(op)) {
        int64_t mallocSize = getMemRefSize(op);
        int64_t mallocOffset = mallocSimul(op, mallocSize);

        auto memRefType = convertToMemRefType(op->getResult(0).getType());
        auto int64Ty = builder.getIntegerType(64);
        auto offsetConst = builder.create<arith::ConstantOp>(loc, int64Ty,
            builder.getI64IntegerAttr(mallocOffset));
        auto mallocSizeConst = builder.create<arith::ConstantOp>(loc, int64Ty,
            builder.getI64IntegerAttr(mallocSize));
        auto memOffsetOp = builder.create<mlir::DNNMemOffsetOp>(loc, memRefType,
            memPoolInitOp->getResult(0), offsetConst, mallocSizeConst);

        offsetConst.getOperation()->moveAfter(op);
        mallocSizeConst.getOperation()->moveAfter(offsetConst);
        memOffsetOp.getOperation()->moveAfter(mallocSizeConst);

        replaceVec.emplace_back(op);
        replaceVec.emplace_back(memOffsetOp->getResult(0).getDefiningOp());

        updateMaxAllocByte(builder, maxAllocByte, memPoolSize);
      } else if (isa<mlir::DNNDeallocOp>(op)) {
        Operation* mallocOpPtr = op->getOperand(0).getDefiningOp();


        int64_t mallocSize = getMemRefSize(mallocOpPtr);

        bool deallocResult = deallocSimul(mallocOpPtr);
        if (!deallocResult) assert(0 && "something wrong when simulating dealloc");

        removeDeallocVec.emplace_back(op);

        updateMaxAllocByte(builder, maxAllocByte, memPoolSize);
      }
    }

    replaceUseForMemOffsetOp(replaceVec);
    eraseAndFreeDeallocVec(removeDeallocVec);
    insertPoolDealloc (builder, memPoolInitOp);
    std::cout << "maxAllocByte : " << maxAllocByte << std::endl;
    clearBlockVectors();
    for (long unsigned int i=0; i<removeDIVVec.size(); i++) { removeDIVVec[i]->erase(); }
    removeDIVVec.clear();
  }
};

std::unique_ptr<Pass> core_dnn::createmallocPoolOptPass() {
  return std::make_unique<mallocPoolOptPass>();
}

