//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNMemcpyOpLowering
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

Value cudaStreamValue;

class DNNMemcpyOpLowering : public ConvertToLLVMPattern {
public:
  DNNMemcpyOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNMemcpyOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::DNNRuntimeAPIRegistry(module, rewriter, inType);

    auto dst = op->getOperand(0);
    auto src = op->getOperand(1);
    auto count = op->getOperand(2);
    int memcpyMode = llvm::dyn_cast<DNNMemcpyOp>(op).getMode();

    auto int8Ty = IntegerType::get(context, 8);
    auto int8PtrTy = LLVM::LLVMPointerType::get(int8Ty);
    auto int32Ty = IntegerType::get(context, 32);

    auto srcElemPtrType = LLVM::LLVMPointerType::get(
        typeConverter->convertType(
          typeConverter->convertType(
            src.getType().cast<mlir::MemRefType>().getElementType()
            )
          ).cast<mlir::Type>()
        );

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedDst = castToLLVMStruct(context, typeConverter, rewriter, loc, dst);
    auto convertedSrc = castToLLVMStruct(context, typeConverter, rewriter, loc, src);

    auto constMemcpyMode = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI64IntegerAttr(memcpyMode));

    auto dstPtr = rewriter.create<LLVM::ExtractValueOp>(loc, srcElemPtrType, convertedDst,
        llvm::ArrayRef<int64_t>{0});
    auto bcDst = rewriter.create<LLVM::BitcastOp>(loc, int8PtrTy, dstPtr);

    auto srcPtr = rewriter.create<LLVM::ExtractValueOp>(loc, srcElemPtrType, convertedSrc,
        llvm::ArrayRef<int64_t>{0});
    auto bcSrc = rewriter.create<LLVM::BitcastOp>(loc, int8PtrTy, srcPtr);

    auto streamStructTy =
      LLVM::LLVMStructType::getOpaque("CUstream_st", context);
    auto streamPtrTy =
      LLVM::LLVMPointerType::get(streamStructTy);

    auto streamLoad = rewriter.create<LLVM::LoadOp>(
        loc, streamPtrTy, cudaStreamValue);

    auto callMemcpy = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CUDA_MEMCPY,
        {bcDst, bcSrc, count, constMemcpyMode, streamLoad});

    rewriter.replaceOp(op, callMemcpy);

    return success();
  }
};

void mlir::populateDNNMemcpyToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value stream) {
  patterns.insert<DNNMemcpyOpLowering>(ctx, typeConverter);
  cudaStreamValue = stream;
}
