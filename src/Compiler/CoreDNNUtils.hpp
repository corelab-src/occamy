/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.hpp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/FileUtilities.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "core-dnn/Compiler/CDCompilerTypes.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/CoreDNNOptions.hpp"
#include "src/Compiler/CoreDNNPasses.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/CDPasses.hpp"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Version/Version.hpp"

namespace core_dnn{
// Return the vendor name if specified during make processing or the default.
std::string getVendorName();

void registerDialects(mlir::MLIRContext &context);
// Process the input model given by its module and context into an output file
// according to the emission target type. Name of the output file can be
// constructed using the getTargetFilename function below.  When  generating
// libraries or jar files, the compiler will link in lightweight runtimes / jar
// files. If these libraries / jar files are not in the system wide directory
// (typically /usr/local/lib), the user can override the default location using
// the ONNX_MLIR_LIBRARY_PATH environment variable.
// Returns 0 on success,OnnxMlirCompilerErrorCodes on failure.
int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    core_dnn::EmissionTargetType emissionTarget,
    HardwareTargetType hardwareTarget);

onnx_mlir::EmissionTargetType emissionCOREToONNX(core_dnn::EmissionTargetType coreEmission);

core_dnn::EmissionTargetType emissionONNXToCORE(onnx_mlir::EmissionTargetType onnxEmission);

} // namespace core_dnn
