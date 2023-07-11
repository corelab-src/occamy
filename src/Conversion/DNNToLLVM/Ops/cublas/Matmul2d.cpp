//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNMatmul2dOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;

class DNNMatmul2dOpLowering : public ConvertToLLVMPattern {
public:
  DNNMatmul2dOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNMatmul2dOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNMatmul2dOp matmul2dOp = dyn_cast<DNNMatmul2dOp>(op);

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

    auto inputA = matmul2dOp.getA();
    auto inputB = matmul2dOp.getB();
    auto outputY = matmul2dOp.getY();

    auto alphaAttr = matmul2dOp.getAlpha().convertToFloat();
    auto betaAttr = matmul2dOp.getBeta().convertToFloat();
    auto transAAttr = matmul2dOp.getTransA();
    auto transBAttr = matmul2dOp.getTransB();

    auto inputADimAttr = matmul2dOp.getDimA();
    auto inputBDimAttr = matmul2dOp.getDimB();
    auto outputYDimAttr = matmul2dOp.getDimY();

    auto inputAMemRefType = inputA.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputAMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputAMemRefShape = inputAMemRefType.getShape();

    // setting alpha, beta and trans attributes
    auto alpha = rewriter.create<LLVM::ConstantOp>(loc, floatTy,
        rewriter.getF32FloatAttr(alphaAttr));
    auto beta = rewriter.create<LLVM::ConstantOp>(loc, floatTy,
        rewriter.getF32FloatAttr(betaAttr));
    auto transA = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(transAAttr));
    auto transB = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(transBAttr));

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedA = castToLLVMStruct(context, typeConverter, rewriter, loc, inputA);
    auto convertedB = castToLLVMStruct(context, typeConverter, rewriter, loc, inputB);
    auto convertedY = castToLLVMStruct(context, typeConverter, rewriter, loc, outputY);

    // Load input and output
    auto extractInputA = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedA,
        llvm::ArrayRef<int64_t>{0});
    auto extractInputB = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedB,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutputY = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedY,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute and padding attr
    int tensorRank = (int)inputAMemRefShape.size();
    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(tensorRank));

    auto inputADim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto inputBDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);
    auto outputYDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < tensorRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

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

      auto outputYDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputYDimAttr[i].cast<IntegerAttr>());
      auto outputYGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputYDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputYDimI, outputYGep);
    }

    // TODO: Complete lowering without the library
    auto callMatmul2d = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, core_dnn::DNNRuntimeAPI::API::CUBLAS_MATMUL_2D,
        {rankConst,
        extractInputA, inputADim,
        extractInputB, inputBDim,
        extractOutputY, outputYDim,
        extractOutputY, outputYDim,
        alpha, beta, transA, transB});

    Value callMatmul2dOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callMatmul2d);

    rewriter.replaceOp(op, callMatmul2dOutput);

    return success();
  }
};

void mlir::populateDNNMatmul2dToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNMatmul2dOpLowering>(ctx, typeConverter);
}
