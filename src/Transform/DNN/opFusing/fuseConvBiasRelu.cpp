//===--------- Start of fuseConvBiasActivPass ----------===//

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
class fuseConvBiasActivPass
    : public PassWrapper<fuseConvBiasActivPass, OperationPass<func::FuncOp>> {

public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getOperation());

    typedef enum {
      NO_FUSING,
      CONVBIAS,
      CONVBIASRELU,
      UNDEF
    } fuseType_t;

    SmallVector<Operation*, 1> opToErase;

    funcOp.walk([&](DNNConvForwardOp op) {
      fuseType_t opFuseType = CONVBIAS;
      DNNAddOp addOp = NULL;
      DNNActivationForwardOp activOp = NULL;
      if (!op->use_empty()) {
        for (Operation* user: op->getUsers()) {
          if (!(addOp = dyn_cast_or_null<DNNAddOp>(user))) {
            opFuseType = NO_FUSING;
            break;
          }
        }
      } else {
        opFuseType = NO_FUSING;
      }

      if(addOp) {
        if (!addOp->use_empty()) {
          for (Operation* user: addOp->getUsers()) {
            activOp = dyn_cast_or_null<DNNActivationForwardOp>(user);
            if (!activOp) {
              opFuseType = CONVBIAS;
              break;
            } else {
              if(activOp.getMode() == 1) { // If the activation is not ReLU
                opFuseType = CONVBIASRELU;
                break;
              } else {
                opFuseType = CONVBIAS;
              }
            }
          }
        } else {
          opFuseType = CONVBIAS;
        }
      }

      if (opFuseType == CONVBIAS || opFuseType == CONVBIASRELU) {
        // From Convolution Op
        mlir::ValueRange convOpOperands = op.getOperands();
        DNNConvForwardOpAdaptor convOpOperandAdaptor(convOpOperands);
        auto input = convOpOperandAdaptor.getX();
        auto weight = convOpOperandAdaptor.getW();
        auto workspace = convOpOperandAdaptor.getWorkspace();
        auto output = convOpOperandAdaptor.getOut();
        auto pads = op.getPads();
        auto strides = op.getStrides();
        auto workspaceSize = op.getWorkspaceSize();
        auto convAlgo = op.getConvAlgorithm();


        auto inputMemRef = convertToMemRefType(input.getType());
        auto weightMemRef = convertToMemRefType(weight.getType());
        auto outputMemRef = convertToMemRefType(output.getType());

        // From Bias Addition Op
        mlir::ValueRange addOpOperands = addOp.getOperands();
        DNNAddOpAdaptor addOpOperandAdaptor(addOpOperands);
        auto bias = addOpOperandAdaptor.getB();
        auto biasMemRef = convertToMemRefType(bias.getType());

        // Compute broadcasting Dim for Input
        SmallVector<int64_t> broadcastedBiasDim;
        broadcastOnlyDimension(&broadcastedBiasDim, outputMemRef, biasMemRef);

        if(opFuseType == CONVBIASRELU) {
          // From Activation Op
          mlir::ValueRange activOpOperands = activOp.getOperands();
          DNNActivationForwardOpAdaptor activOpOperandAdaptor(activOpOperands);
          auto activResultMalloc = activOpOperandAdaptor.getY();
          auto activResultMallocOp = activResultMalloc.getDefiningOp();
          auto activResultMemRef = convertToMemRefType(activResultMalloc.getType());

          auto fusedOp = builder.create<DNNConvBiasActivForwardOp>(
              funcOp.getLoc(), activResultMemRef,
              input, builder.getI64ArrayAttr(inputMemRef.getShape()),
              weight, builder.getI64ArrayAttr(weightMemRef.getShape()),
              bias, builder.getI64ArrayAttr(broadcastedBiasDim),
              workspace, builder.getI64IntegerAttr(workspaceSize), pads, strides,
              builder.getI64IntegerAttr(1), // DNN_ACTIVATION_RELU
              builder.getI64IntegerAttr(convAlgo), activResultMalloc);

          fusedOp.getOperation()->moveAfter(activResultMallocOp);
          activOp->replaceAllUsesWith(fusedOp);

          auto dummyMalloc = op.getOut().getDefiningOp();
          DNNDeallocOp dummyDealloc;
          for (Operation* user: dummyMalloc->getUsers()) {
            if (isa<DNNDeallocOp>(user)) {
              dummyDealloc = dyn_cast_or_null<DNNDeallocOp>(user);
            }
          }

          opToErase.insert(opToErase.begin(), dummyMalloc);
          if (dummyDealloc)
            opToErase.insert(opToErase.begin(), dummyDealloc);
          opToErase.insert(opToErase.begin(), op);
          opToErase.insert(opToErase.begin(), addOp);
          opToErase.insert(opToErase.begin(), activOp);
        } else if (opFuseType == CONVBIAS) {
          auto convResultMalloc = convOpOperandAdaptor.getOut();
          auto convResultMallocOp = convResultMalloc.getDefiningOp();
          auto resultMemRef = convertToMemRefType(convResultMalloc.getType());

          auto fusedOp = builder.create<DNNConvBiasActivForwardOp>(
              funcOp.getLoc(), resultMemRef,
              input, builder.getI64ArrayAttr(inputMemRef.getShape()),
              weight, builder.getI64ArrayAttr(weightMemRef.getShape()),
              bias, builder.getI64ArrayAttr(broadcastedBiasDim),
              workspace, builder.getI64IntegerAttr(workspaceSize), pads, strides,
              builder.getI64IntegerAttr(5), // DNN_ACTIVATION_IDENTITY
              builder.getI64IntegerAttr(convAlgo), convResultMalloc);

          fusedOp.getOperation()->moveAfter(convResultMallocOp);
          addOp->replaceAllUsesWith(fusedOp);

          opToErase.insert(opToErase.begin(), op);
          opToErase.insert(opToErase.begin(), addOp);
        }
      }
    });

    for (long unsigned int i=0; i<opToErase.size(); i++) {
      opToErase[i]->erase();
    }
  }
};

std::unique_ptr<Pass> core_dnn::createfuseConvBiasActivPass() {
  return std::make_unique<fuseConvBiasActivPass>();
}

