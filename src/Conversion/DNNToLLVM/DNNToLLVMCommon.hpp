//===------ DNNToLLVM.hpp - Lowering from KRNL+Affine+DNN+Std to LLVM -------===//
//
// Copyright 2021 Yonsei CORELAB.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#ifndef DNN_TO_LLVM_H
#define DNN_TO_LLVM_H

#include "mlir/Support/LLVM.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Value.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Pass/CDPasses.hpp"
#include "src/Support/KrnlSupport.hpp"

// #include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
using namespace mlir;

void generateDNNHandle(MLIRContext *context, mlir::ModuleOp &m, Value &cudnnHandle);
void generateCUDAStream(MLIRContext *context, mlir::ModuleOp &m, Value &cudaStreamValue);

namespace mlir {

class MLIRContext;
class LLVMTypeConverter;
class RewritePatternSet;

// Insert the insertValueOp for op output shape information. After make the
// insertValueOps it returns the last insertValueOp for usage of original op.
Value insertAndReturnOutputShapeInfo (
    MLIRContext* context, Location loc, TypeConverter* typeConverter,
    ConversionPatternRewriter &rewriter, Value cudnnOutput, Value llvmOp);

// Insert unrealized conversion cast op to convert memref to llvm struct type.
Value castToLLVMStruct(MLIRContext *context, TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location &loc, Value v);

void populateDNNToLLVMConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle, Value stream);

// ------------------- CUDA ---------------------//
void populateDNNMallocToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNMemPoolInitToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNMemOffsetToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNDeallocToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNCastToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNClipToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNMemcpyToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value stream);
void populateDNNConcatToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNReciprocalToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNNegativeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNErfToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNFlattenToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNReshapeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNSqueezeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNUnsqueezeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNTransposeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNExpandToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNGatherToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNNonZeroToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNPowToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNMatmulNdToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNPReluToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNSoftmaxToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
void populateDNNLeakyReluToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);

// ------------------ DNN --------------------//
void populateDNNConvForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNActivationForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNReduceToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNMaxPoolToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNAveragePoolToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNAddToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNMulToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);
void populateDNNSqrtToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);

void populateDNNConvBiasActivForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle);

// ------------------ cuBLAS --------------------//
void populateDNNMatmul2dToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter);
} // namespace mlir

#endif // DNN_TO_LLVM_H
