//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNTransposeOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNTransposeOpLowering : public ConvertToLLVMPattern {
public:
  DNNTransposeOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNTransposeOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNTransposeOp transposeOp = dyn_cast<DNNTransposeOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::DNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);

    auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(0));

    auto input = transposeOp.getX();
    auto output = transposeOp.getY();
    auto inputDimAttr = transposeOp.getDimX();
    auto outputDimAttr = transposeOp.getDimY();
    auto permAttr = transposeOp.getPerm();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    int inputRank = inputMemRefShape.size();
    assert((((inputRank>1)&&(inputRank<=4))||(inputRank==6)) && "Only support 2, 3, 4, 6D tensor transpose");

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputRank));
    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto permArg = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < inputRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          inputDimAttr[i].cast<IntegerAttr>());
      auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);

      auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputDimAttr[i].cast<IntegerAttr>());
      auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);

      auto permArgI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          permAttr[i].cast<IntegerAttr>());
      auto permArgGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, permArg,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, permArgI, permArgGep);
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    Value callTranspose;
    if (inputRank == 2) {
      if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_2D_I64,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_2D_F32,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      }
    } else if (inputRank == 3) {
      if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_3D_I64,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_3D_F32,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      }
    } else if (inputRank == 4) {
      if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_4D_I64,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_4D_F32,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      }
    } else if (inputRank == 6) {
      if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_6D_I64,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
        callTranspose = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
            apiRegistry, DNNRuntimeAPI::API::CUDA_TRANSPOSE_6D_F32,
            {extractInput, inputDim, extractOutput, outputDim, permArg, rankConst});
      }
    } else {
      return emitError(loc, "TransposeToLLVM: Unsupported input dimension (Only support 2, 3, 4D Tensor)");
    }

    Value callTransposeOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callTranspose);

    rewriter.replaceOp(op, callTransposeOutput);

    return success();
  }
};

void mlir::populateDNNTransposeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNTransposeOpLowering>(ctx, typeConverter);
}
