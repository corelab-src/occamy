//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNMatmulNdOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNMatmulNdOpLowering : public ConvertToLLVMPattern {
public:
  DNNMatmulNdOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNMatmulNdOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNMatmulNdOp matmulndOp = dyn_cast<DNNMatmulNdOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    DNNMatmulNdOpAdaptor operandAdaptor(operands);
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

    auto inputA = matmulndOp.getA();
    auto inputB = matmulndOp.getB();
    auto output = matmulndOp.getY();
    auto inputADimAttr = matmulndOp.getDimA();
    auto inputBDimAttr = matmulndOp.getDimB();
    auto outputDimAttr = matmulndOp.getDimY();

    auto memRefType = inputA.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);

    int inputARank = inputADimAttr.size();
    int inputBRank = inputBDimAttr.size();
    int outputRank = outputDimAttr.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedA = castToLLVMStruct(context, typeConverter, rewriter, loc, inputA);
    auto convertedB = castToLLVMStruct(context, typeConverter, rewriter, loc, inputB);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and weight
    auto extractInputA = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedA,
        llvm::ArrayRef<int64_t>{0});
    auto extractInputB = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedB,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto inputARankConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(inputARank));
    auto inputADim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, inputARankConst, 0);
    for (int i = 0; i < inputARank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto inputADimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          inputADimAttr[i].cast<IntegerAttr>());
      auto inputAGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputADim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, inputADimI, inputAGep);
    }

    auto inputBRankConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(inputBRank));
    auto inputBDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, inputBRankConst, 0);
    for (int i = 0; i < inputBRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto inputBDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          inputBDimAttr[i].cast<IntegerAttr>());
      auto inputBGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputBDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, inputBDimI, inputBGep);
    }

    auto outputRankConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(outputRank));
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, outputRankConst, 0);
    for (int i = 0; i < outputRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputDimAttr[i].cast<IntegerAttr>());
      auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
    }

    // TODO: Complete lowering without the library
    auto callMatmulNd = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CUDA_MATMUL_ND,
        {extractInputA, inputADim, inputARankConst,
        extractInputB, inputBDim, inputBRankConst,
        extractOutput, outputDim, outputRankConst});

    Value callMatmulNdOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callMatmulNd);

    rewriter.replaceOp(op, callMatmulNdOutput);

    return success();
  }
};

void mlir::populateDNNMatmulNdToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNMatmulNdOpLowering>(ctx, typeConverter);
}
