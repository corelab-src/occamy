//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNSqrtOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

Value cudnnSqrtHandle;

class DNNSqrtOpLowering : public ConvertToLLVMPattern {
public:
  DNNSqrtOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNSqrtOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNSqrtOp sqrtOp = dyn_cast<DNNSqrtOp>(op);

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

    auto inputA = sqrtOp.getX();
    auto output = sqrtOp.getResult();
    auto outputDimAttr = sqrtOp.getDimResult();

    auto memRefType = inputA.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto memRefShape = memRefType.getShape();

    int outputRank = outputDimAttr.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, inputA);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and weight
    auto extractInputA = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    // Due to cudnnSetTensorNdDescriptor restriction, only 4D tensors can be computed.
    // For more info : https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor
    int adjustedRank = 0;
    if(outputRank <= 3) {
      adjustedRank = 4;
    } else {
      adjustedRank = outputRank;
    }

    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(adjustedRank));

    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < adjustedRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      if(i < outputRank) {
        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            outputDimAttr[i].cast<IntegerAttr>());
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      } else {
        auto constOne= rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(1));

        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, constOne, outputGep);
      }
    }

    auto handleStructTy =
      LLVM::LLVMStructType::getOpaque("cudnnContext", context);
    auto handlePtrTy =
      LLVM::LLVMPointerType::get(handleStructTy);

    auto handleLoad = rewriter.create<LLVM::LoadOp>(
        loc, handlePtrTy, cudnnSqrtHandle);

    auto constBias = rewriter.create<LLVM::ConstantOp>(loc, floatTy,
        rewriter.getF32FloatAttr(0.f));

    auto opModeConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(4)/* DNN_OP_TENSOR_SQRT*/);

    // TODO: Complete lowering without the library
    auto callSqrt = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CSRC_TENSOR_OP,
        {handleLoad, opModeConst,
        extractInputA, outputDim, rankConst,
        extractInputA, outputDim, rankConst,
        extractOutput, outputDim, rankConst,
        constBias});

    Value callSqrtOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callSqrt);

    rewriter.replaceOp(op, callSqrtOutput);

    return success();
  }
};

void mlir::populateDNNSqrtToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle) {
  patterns.insert<DNNSqrtOpLowering>(ctx, typeConverter);
  cudnnSqrtHandle = handle;
}
