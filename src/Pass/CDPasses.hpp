#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace core_dnn {

/// Pass for lowering ONNX dialect to DNN dialect
std::unique_ptr<Pass> createConvertONNXToDNNPass();

/// Pass for lowering DNN dialect to LLVM dialect
std::unique_ptr<Pass> createConvertDNNToLLVMPass();

/// Pass for transforming DNN dialect
std::unique_ptr<Pass> createONNXConstantHoistingPass();
std::unique_ptr<Pass> createONNXConstantAtUsePass();
std::unique_ptr<Pass> createFuncOpArgumentToDNNPass();
std::unique_ptr<Pass> createDNNDeallocOptPass();
std::unique_ptr<Pass> createeraseDummyConstantsPass();
std::unique_ptr<Pass> createmallocPoolOptPass();
std::unique_ptr<Pass> createfuseConvBiasActivPass();
}

