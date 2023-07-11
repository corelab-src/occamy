/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ DNNRuntimeAPI.cpp - Implementation of Runtime API -----------===//
//
// This file implements the Runtime API.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace core_dnn {
//===----------------------------------------------------------------------===//
// DNNRuntimeAPI
//===----------------------------------------------------------------------===//

void DNNRuntimeAPI::declareAPI(ModuleOp &module, OpBuilder &builder) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::LLVMBuilder> create(
      builder, module.getLoc());
  symbolRef =
      create.llvm.getOrInsertSymbolRef(module, name, outputTy, inputTys);
}

// Call a registered API, return the return SSA values if only one result is
// returned, otherwise return nullptr.
Value DNNRuntimeAPI::callApi(OpBuilder &builder, Location loc,
    const DNNRuntimeAPIRegistry &registry, API apiId, ArrayRef<Value> params) {
  onnx_mlir::MultiDialectBuilder<onnx_mlir::LLVMBuilder> create(builder, loc);
  // To be used as parameters in LLVM::CallOp, voidTy must be converted
  // to empty list to avoid emission of an SSA value with voidTy. However,
  // we still keep using LLVM voidTy (as opposed to empty list) when recording
  // API function signatures in API registry because when declaring API
  // functions in LLVM IR, the correct way to indicate an output type for
  // "void" is still LLVM voidTy. Relevant discussion thread:
  // https://github.com/onnx/onnx-mlir/issues/255.
  SmallVector<Type, 1> outputTys;
  const DNNRuntimeAPI &runtimeAPI = registry.getAPI(apiId);
  auto outputTy = runtimeAPI.outputTy;
  if (!outputTy.isa<LLVM::LLVMVoidType>())
    outputTys.emplace_back(outputTy);
  return create.llvm.call(ArrayRef<Type>(outputTys),
      registry.getAPI(apiId).symbolRef, ArrayRef<Value>(params));
}

//===----------------------------------------------------------------------===//
// DNNRuntimeAPIRegistry
//===----------------------------------------------------------------------===//

DNNRuntimeAPIRegistry::~DNNRuntimeAPIRegistry() {}

DNNRuntimeAPIRegistry::DNNRuntimeAPIRegistry(ModuleOp &module, OpBuilder &builder,
    mlir::Type inType)
    : registry() {
  MLIRContext *context = module.getContext();
  auto voidTy = LLVM::LLVMVoidType::get(context);
  auto int8Ty = IntegerType::get(context, 8);
  auto int8PtrTy = LLVM::LLVMPointerType::get(int8Ty);
  auto int8PtrPtrTy = LLVM::LLVMPointerType::get(int8PtrTy);
  auto opaquePtrTy = LLVM::LLVMPointerType::get(int8Ty);
  auto opaquePtrPtrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
  auto int32Ty = IntegerType::get(context, 32);
  auto int64Ty = IntegerType::get(context, 64);
  auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
  auto int64PtrPtrTy = LLVM::LLVMPointerType::get(int64PtrTy);
  mlir::Type floatTy = FloatType::getF32(context);
  if (inType.isF64())
    floatTy = FloatType::getF64(context);
  auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);
  auto floatPtrPtrTy = LLVM::LLVMPointerType::get(floatPtrTy);

  auto handleStructTy =
    LLVM::LLVMStructType::getOpaque("cudnnContext", context);
  auto handlePtrTy =
    LLVM::LLVMPointerType::get(handleStructTy);
  auto handlePtrAddrTy =
    LLVM::LLVMPointerType::get(handlePtrTy);

  auto streamStructTy =
    LLVM::LLVMStructType::getOpaque("CUstream_st", context);
  auto streamPtrTy =
    LLVM::LLVMPointerType::get(streamStructTy);
  auto streamPtrAddrTy =
    LLVM::LLVMPointerType::get(streamPtrTy);


  // Declare API type as an enum value, its string name and an LLVM Type
  // specifying its signature.
  // clang-format off
  using API = DNNRuntimeAPI::API;
  #include "src/Conversion/DNNToLLVM/DNNAPISpecs.inc"
  // clang-format on

  // Declare APIs in the current module and build an API registry mapping api
  // identities to a symbol reference to the API function.
  for (auto &apiSpec : DNNRuntimeAPISpecs) {
    apiSpec.declareAPI(module, builder);
    registry.emplace(apiSpec.id, apiSpec);
  }
}

} // namespace core_dnn
