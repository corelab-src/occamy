#pragma once
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Accelerators/Accelerator.hpp"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace core_dnn {
  // CoreDNN optimization options
  extern llvm::cl::OptionCategory CoreDnnOptions;

  // the option is used in this file, so defined here
  extern llvm::cl::opt<bool> onnxConstHoisting;
  extern llvm::cl::opt<bool> onnxConstAtUse;
  extern llvm::cl::opt<bool> dnnKernelFusion;
  extern llvm::cl::opt<bool> dnnDeallocOpt;
  extern llvm::cl::opt<bool> dnnmallocPoolOpt;
}
