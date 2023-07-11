//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNActivationForwardOpLowering
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

Value cudnnActivationHandle;

class DNNActivationForwardOpLowering : public ConvertToLLVMPattern {
public:
  DNNActivationForwardOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNActivationForwardOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNActivationForwardOp activatefwdOp = dyn_cast<DNNActivationForwardOp>(op);

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

    auto four32 = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(4));

    auto input = activatefwdOp.getX();
    auto output = activatefwdOp.getY();
    auto inputDimAttr = activatefwdOp.getDimX();
    int activeType = llvm::dyn_cast<DNNActivationForwardOp>(op).getMode();

    auto memRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto memRefShape = memRefType.getShape();

    int inputRank = inputDimAttr.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and weight
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);

    for (int i = 0; i < 4; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      if(i < inputRank) {
        auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            inputDimAttr[i].cast<IntegerAttr>());
        auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);
      } else {
        auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(1));
        auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);
      }
    }

    // handle load operation
    auto handleStructTy =
      LLVM::LLVMStructType::getOpaque("cudnnContext", context);
    auto handlePtrTy =
      LLVM::LLVMPointerType::get(handleStructTy);

    auto handleLoad = rewriter.create<LLVM::LoadOp>(
        loc, handlePtrTy, cudnnActivationHandle);

    // Activation type constant op
    auto activateTyOp = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(activeType));

    // Call C coded library (../csrc/DNNActivateFunc.cpp)
    // TODO: Complete lowering without the library
    auto callActive = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CSRC_ACTIVEFWD,
        {handleLoad, extractInput, inputDim, activateTyOp, extractOutput});

    Value callActiveOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callActive);

    rewriter.replaceOp(op, callActiveOutput);

    return success();
  }
};

void mlir::populateDNNActivationForwardToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle) {
  patterns.insert<DNNActivationForwardOpLowering>(ctx, typeConverter);
  cudnnActivationHandle = handle;
}
