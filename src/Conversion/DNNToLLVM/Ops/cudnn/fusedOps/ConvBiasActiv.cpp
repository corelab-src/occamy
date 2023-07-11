//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNConvBiasActivOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

Value cudnnConvBiasActivHandle;

class DNNConvBiasActivOpLowering : public ConvertToLLVMPattern {
public:
  DNNConvBiasActivOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNConvBiasActivForwardOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNConvBiasActivForwardOp convbiasactivOp = dyn_cast<DNNConvBiasActivForwardOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::DNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);

    auto two32 = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(2));
    auto four32 = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(4));

    auto input = convbiasactivOp.getX();
    auto weight = convbiasactivOp.getW();
    auto workspace= convbiasactivOp.getWorkspace();
    auto bias = convbiasactivOp.getB();

    auto inputDimAttr = convbiasactivOp.getDimX();
    auto weightDimAttr = convbiasactivOp.getDimW();
    auto workspaceSizeAttr = convbiasactivOp.getWorkspaceSize();
    auto biasDimAttr = convbiasactivOp.getDimB();

    auto padsAttr = convbiasactivOp.getPads();
    auto stridesAttr = convbiasactivOp.getStrides();
    auto output = op->getOperand(4);

    auto memRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto memRefShape = memRefType.getShape();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedWeight = castToLLVMStruct(context, typeConverter, rewriter, loc, weight);
    auto convertedBias = castToLLVMStruct(context, typeConverter, rewriter, loc, bias);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);
    auto convertedWorkspace = castToLLVMStruct(context, typeConverter, rewriter, loc, workspace);

    // Load input and weight
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractWeight = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedWeight,
        llvm::ArrayRef<int64_t>{0});
    auto extractBias = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedBias,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});
    auto extractWorkspace = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedWorkspace,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto weightDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto biasDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    for (int i = 0; i < 4; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          inputDimAttr[i].cast<IntegerAttr>());
      auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);

      auto weightDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          weightDimAttr[i].cast<IntegerAttr>());
      auto weightGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, weightDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, weightDimI, weightGep);

      auto biasDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          biasDimAttr[i].cast<IntegerAttr>());
      auto biasGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, biasDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, biasDimI, biasGep);
    }

    // Create integer array from pad, stride attribute
    auto pads = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, two32, 0);
    auto strides = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, two32, 0);
    for (int i = 0; i < 2; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto padsI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          padsAttr[i].cast<IntegerAttr>());
      auto padsGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, pads,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, padsI, padsGep);

      auto stridesI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          stridesAttr[i].cast<IntegerAttr>());
      auto stridesGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, strides,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, stridesI, stridesGep);
    }

    auto handleStructTy =
      LLVM::LLVMStructType::getOpaque("cudnnContext", context);
    auto handlePtrTy =
      LLVM::LLVMPointerType::get(handleStructTy);

    auto handleLoad = rewriter.create<LLVM::LoadOp>(
        loc, handlePtrTy, cudnnConvBiasActivHandle);

    // DNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM TODO: find algorithm in static time! not just 1
    auto algoConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(1));

    // DNN Workspace Size
    auto workspaceSizeConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(workspaceSizeAttr));

    auto activModeConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(convbiasactivOp.getActivMode()));

    // TODO: Complete lowering without the library
    auto callConv = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CSRC_CONV_BIAS_RELUFWD,
        {handleLoad, extractInput, inputDim,
        extractWeight, weightDim,
        extractBias, biasDim,
        extractWorkspace, workspaceSizeConst, algoConst,
        pads, strides, activModeConst, extractOutput});

    Value callConvOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callConv);

    rewriter.replaceOp(op, callConvOutput);

    return success();
  }
};

void mlir::populateDNNConvBiasActivForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle) {
  patterns.insert<DNNConvBiasActivOpLowering>(ctx, typeConverter);
  cudnnConvBiasActivHandle = handle;
}
