/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DNNRuntimeAPI.hpp - Declaration of the Runtime API ----------===//
//
// This file declare the Runtime API the compiler can use.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

#include <string>

namespace core_dnn {

class DNNRuntimeAPIRegistry;

/// \class DNNRuntimeAPI
/// Represents a Runtime API callable by the compiler.
/// Instances of this class can only be created by the DNNRuntimeAPIRegistry
/// class.
class DNNRuntimeAPI final {
  friend class DNNRuntimeAPIRegistry;

public:
  // Enumerate the runtime functions.
  enum class API {
    // CUBLAS
    CUBLAS_MATMUL_2D,

    // CUDA Custom kernels
    CUDA_CAST,
    CUDA_CLIP,
    CUDA_CONCAT_I64,
    CUDA_CONCAT_F32,
    DNN_DEALLOC,
    CUDA_GATHER_I64,
    CUDA_GATHER_F32,
    CUDA_ERF,
    CUDA_EXPAND,
    CUDA_FLATTEN,
    CUDA_LEAKYRELU,
    CUDA_MALLOC,
    CUDA_MATMUL_ND,
    CUDA_MEMCPY,
    CUDA_MEMOFFSET,
    CUDA_MEMPOOLINIT,
    CUDA_NEGATIVE,
    CUDA_NONZERO_I64,
    CUDA_NONZERO_F32,
    CUDA_POW,
    CUDA_PRELU,
    CUDA_RECIPROCAL,
    CUDA_RESHAPE,
    CSRC_SOFTMAX,
    CUDA_SQUEEZE_I64,
    CUDA_SQUEEZE_F32,
    CUDA_TRANSPOSE_2D_I64,
    CUDA_TRANSPOSE_2D_F32,
    CUDA_TRANSPOSE_3D_I64,
    CUDA_TRANSPOSE_3D_F32,
    CUDA_TRANSPOSE_4D_I64,
    CUDA_TRANSPOSE_4D_F32,
    CUDA_TRANSPOSE_6D_I64,
    CUDA_TRANSPOSE_6D_F32,
    CUDA_UNSQUEEZE_I64,
    CUDA_UNSQUEEZE_F32,

    // DNN utilizing kernels
    CUDNN_CREATE,
    CUDNN_DESTROY,
    STREAM_CREATE,
    STREAM_DESTROY,

    CSRC_ACTIVEFWD,
    CSRC_TENSOR_OP,
    CSRC_AVERAGEPOOL,
    CSRC_CONVFWD,
    CSRC_MAXPOOL,
    CSRC_REDUCE,

    CSRC_CONV_BIAS_RELUFWD,
  };

  // Call the runtime API identified by \p apiId, return the SSA value
  // representing the call.
  static mlir::Value callApi(mlir::OpBuilder &builder, mlir::Location loc,
      const DNNRuntimeAPIRegistry &registry, API apiId,
      llvm::ArrayRef<mlir::Value> params);

private:
  DNNRuntimeAPI(API id, const std::string &name, mlir::Type outputTy,
      llvm::ArrayRef<mlir::Type> inputTys)
      : id(id), name(name), outputTy(outputTy),
        inputTys(inputTys.begin(), inputTys.end()) {}

  // Inject the declaration for this runtime API into the given module (unless a
  // declaration exists already).
  void declareAPI(mlir::ModuleOp &module, mlir::OpBuilder &builder);

  static mlir::FlatSymbolRefAttr getOrInsertExternFunc(llvm::StringRef funcName,
      mlir::ModuleOp module, mlir::Type funcType, mlir::OpBuilder &builder);

private:
  API id;
  std::string name;
  mlir::Type outputTy;
  llvm::SmallVector<mlir::Type, 4> inputTys;
  mlir::FlatSymbolRefAttr symbolRef;
};

/// \class DNNRuntimeAPIRegistry
/// Holds the registry for the Runtime APIs the compiler can use.
class DNNRuntimeAPIRegistry final {
public:
  using ApiRegistry = std::map<DNNRuntimeAPI::API, DNNRuntimeAPI>;

  DNNRuntimeAPIRegistry(mlir::ModuleOp &module, mlir::OpBuilder &builder, mlir::Type inType);
  ~DNNRuntimeAPIRegistry();

  static const DNNRuntimeAPIRegistry build(
      mlir::ModuleOp &module, mlir::OpBuilder &builder);

  const DNNRuntimeAPI &getAPI(DNNRuntimeAPI::API apiId) const {
    assert((registry.find(apiId) != registry.end()) &&
           "apiId not found in registry");
    return registry.at(apiId);
  }

private:
  ApiRegistry registry;
};

} // namespace core_dnn
