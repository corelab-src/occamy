//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNAddOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

Value cudnnAddHandle;

class DNNAddOpLowering : public ConvertToLLVMPattern {
public:
  DNNAddOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNAddOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNAddOp addOp = dyn_cast<DNNAddOp>(op);

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
    SmallVector<mlir::Type, 4> outputTys(
        {floatPtrTy, floatPtrTy, int64Ty, int64ArrayTy, int64ArrayTy});
    auto returnTy = LLVM::LLVMStructType::getLiteral(context, outputTys);

    auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(0));

    auto inputA = addOp.getA();
    auto inputB = addOp.getB();
    auto output = addOp.getResult();
    auto inputADimAttr = addOp.getDimA();
    auto inputBDimAttr = addOp.getDimB();
    auto outputDimAttr = addOp.getDimResult();
    auto biasBAttr = addOp.getBiasB().convertToFloat();

    auto memRefType = inputA.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);

    int inputARank = inputADimAttr.size();
    int outputRank = outputDimAttr.size();

    // Load input and weight
    Value convertedInputA = castToLLVMStruct(context, typeConverter, rewriter, loc, inputA);
    Value convertedInputB = castToLLVMStruct(context, typeConverter, rewriter, loc, inputB);
    Value convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    auto extractInputA = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInputA,
        llvm::ArrayRef<int64_t>{0});
    auto extractInputB = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInputB,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    // Due to cudnnSetTensorNdDescriptor restriction, only 4D tensors can be computed.
    // For more info : https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetTensorNdDescriptor
    int inputRank = 0;
    if(inputARank <= 3) {
      inputRank = 4;
    } else {
      inputRank = inputARank;
    }

    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(inputRank));

    auto inputADim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto inputBDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < inputRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      if(i < inputARank) {
        auto inputADimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            inputADimAttr[i].cast<IntegerAttr>());
        auto inputAGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputADim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputADimI, inputAGep);

        auto inputBDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            inputBDimAttr[i].cast<IntegerAttr>());
        auto inputBGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputBDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputBDimI, inputBGep);

        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            outputDimAttr[i].cast<IntegerAttr>());
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      } else {
        auto constOne= rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(1));

        auto inputAGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputADim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, constOne, inputAGep);

        auto inputBGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputBDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, constOne, inputBGep);

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
        loc, handlePtrTy, cudnnAddHandle);

    auto constBias = rewriter.create<LLVM::ConstantOp>(loc, floatTy,
        rewriter.getF32FloatAttr(biasBAttr));

    auto opModeConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(0)/* DNN_OP_TENSOR_ADD*/);

    // TODO: Complete lowering without the library
    auto callAdd = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CSRC_TENSOR_OP,
        {handleLoad, opModeConst,
        extractInputA, inputADim, rankConst,
        extractInputB, inputBDim, rankConst,
        extractOutput, outputDim, rankConst,
        constBias});

    Value callAddOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, output, callAdd);

    rewriter.replaceOp(op, callAddOutput);
        // .getResult());

    return success();
  }
};

void mlir::populateDNNAddToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle) {
  patterns.insert<DNNAddOpLowering>(ctx, typeConverter);
  cudnnAddHandle = handle;
}
