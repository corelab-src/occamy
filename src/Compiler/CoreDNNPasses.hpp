#pragma once
#include "core-dnn/Compiler/CDCompilerTypes.h"
#include "mlir/Pass/PassManager.h"

namespace core_dnn {

void addONNXToDNNPasses(mlir::PassManager &pm);
void addDNNToLLVMPasses(mlir::PassManager &pm);
void addONNXToNPUPasses(mlir::PassManager &pm);
void addNPUToLLVMPasses(mlir::PassManager &pm);

} // namespace core_dnn
