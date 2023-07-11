//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNUnsqueezeOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNUnsqueezeOpLowering : public ConvertToLLVMPattern {
public:
  DNNUnsqueezeOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNUnsqueezeOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNUnsqueezeOp unsqueezeOp = dyn_cast<DNNUnsqueezeOp>(op);

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
    auto four32 = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(4));

    auto input = unsqueezeOp.getX();
    auto output = unsqueezeOp.getY();
    auto inputDimAttr = unsqueezeOp.getDimInput();
    auto outputDimAttr = unsqueezeOp.getDimOutput();
    auto axesAttr = unsqueezeOp.getAxes();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto axes = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);

    int inputRank = inputMemRefShape.size();
    int outputRank = outputDimAttr.size();
    int axesRank = axesAttr.size();

    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputRank));

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
            rewriter.getI64IntegerAttr(-1));
        auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);
      }

      if(i < outputRank) {
        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            outputDimAttr[i].cast<IntegerAttr>());
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      } else {
        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(-1));
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      }

      if(i < axesRank) {
        auto axesI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            axesAttr[i].cast<IntegerAttr>());
        auto axesGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, axes,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, axesI, axesGep);
      } else {
        auto axesI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(-1));
        auto axesGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, axes,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, axesI, axesGep);
      }
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    Value callUnsqueeze;
    if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
      callUnsqueeze = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, DNNRuntimeAPI::API::CUDA_UNSQUEEZE_I64,
          {extractInput, inputDim, extractOutput, outputDim, axes, rankConst});
    } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
      callUnsqueeze = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, DNNRuntimeAPI::API::CUDA_UNSQUEEZE_F32,
          {extractInput, inputDim, extractOutput, outputDim, axes, rankConst});
    }

    Value callUnsqueezeOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callUnsqueeze);

    rewriter.replaceOp(op, callUnsqueezeOutput);

    return success();
  }
};

void mlir::populateDNNUnsqueezeToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNUnsqueezeOpLowering>(ctx, typeConverter);
}
