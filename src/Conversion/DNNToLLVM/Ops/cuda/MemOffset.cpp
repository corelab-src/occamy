//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNMemOffsetOpLowering
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNMemOffsetOpLowering : public ConvertToLLVMPattern {
public:
  DNNMemOffsetOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNMemOffsetOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::DNNRuntimeAPIRegistry(module, rewriter, inType);

    auto base = op->getOperand(0);
    auto offset = op->getOperand(1);
    auto mallocSize = op->getOperand(2);

    auto voidTy = LLVM::LLVMVoidType::get(context);
    auto opaquePtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto opaquePtrPtrTy = LLVM::LLVMPointerType::get(opaquePtrTy);
    auto int8Ty = IntegerType::get(context, 8);
    auto int8PtrTy = LLVM::LLVMPointerType::get(int8Ty);
    auto int8PtrPtrTy = LLVM::LLVMPointerType::get(int8PtrTy);
    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);

    auto memRefType = op->getResult(0).getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto memRefShape = memRefType.getShape();
    auto llvmelemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto allocPtrType = LLVM::LLVMPointerType::get(llvmelemType);
    auto allocPtrPtrType = LLVM::LLVMPointerType::get(allocPtrType);

    auto srcElemPtrType = LLVM::LLVMPointerType::get(
        typeConverter->convertType(
          typeConverter->convertType(
            base.getType().cast<mlir::MemRefType>().getElementType()
            )
          ).cast<mlir::Type>()
        );

    // common constantOp
    auto zero64 = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(0));
    auto one32 = rewriter.create<LLVM::ConstantOp>(
        loc, int32Ty, rewriter.getI32IntegerAttr(1));

    auto devPtr = rewriter.create<LLVM::AllocaOp>(loc, allocPtrPtrType, one32, 0);
    auto devVoidPtr = rewriter.create<LLVM::BitcastOp>(loc, opaquePtrPtrTy, devPtr);

    auto convertedBase = castToLLVMStruct(context, typeConverter, rewriter, loc, base);
    auto basePtr = rewriter.create<LLVM::ExtractValueOp>(loc, srcElemPtrType, convertedBase,
        llvm::ArrayRef<int64_t>{0});
    auto bcBase = rewriter.create<LLVM::BitcastOp>(loc, int8PtrTy, basePtr);

    int64_t offsetInt =
      dyn_cast<arith::ConstantOp>(offset.getDefiningOp()).getValue().cast<IntegerAttr>().getInt();
    auto offsetConst = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(offsetInt));

    int64_t mallocSizeInt =
      dyn_cast<arith::ConstantOp>(mallocSize.getDefiningOp()).getValue().cast<IntegerAttr>().getInt();
    auto mallocSizeConst = rewriter.create<LLVM::ConstantOp>(
        loc, int64Ty, rewriter.getI64IntegerAttr(mallocSizeInt));

    auto callMemOffset = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CUDA_MEMOFFSET,
        {devVoidPtr, bcBase, offsetConst, mallocSizeConst});

    auto loadPtr = rewriter.create<LLVM::LoadOp>(loc, allocPtrType, devPtr);

    Value callMemOffsetOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), loadPtr);

    rewriter.replaceOp(op, callMemOffsetOutput);

    return success();
  }
};

void mlir::populateDNNMemOffsetToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNMemOffsetOpLowering>(ctx, typeConverter);
}
