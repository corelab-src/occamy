//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNAveragePoolOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"


#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

Value cudnnAveragePoolHandle;

class DNNAveragePoolOpLowering : public ConvertToLLVMPattern {
public:
  DNNAveragePoolOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNAveragePoolOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNAveragePoolOp averagePoolOp = dyn_cast<DNNAveragePoolOp>(op);

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

    auto input = op->getOperand(0);
    auto output = op->getOperand(1);
    auto inputDimAttr = averagePoolOp.getDimInput();
    auto outputDimAttr = averagePoolOp.getDimOutput();
    auto kernelShapes = averagePoolOp.getKernelShape();
    auto paddings = averagePoolOp.getPads();
    auto strides = averagePoolOp.getStrides();

    if (kernelShapes.size() != 2)
      return emitError(loc, "AveragePoolToLLVM: kernelShapes: UNKNOWN KERNEL SHAPE! (Now only support 2D kernel shape)");
    if (paddings.size() != 4)
      return emitError(loc, "AveragePoolToLLVM: paddings: Now only support 4 direction paddings (h1, h2, w1, w2)");
    if (strides.size() != 2)
      return emitError(loc, "AveragePoolToLLVM: strides: Now only support 2 direction stride (h,w)");

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute and padding attr
    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto paddingArgs = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    for (int i = 0; i < 4; i++) {
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

      auto paddingArgsI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          paddings[i].cast<IntegerAttr>());
      auto paddingGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, paddingArgs,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, paddingArgsI, paddingGep);
    }

    // Create integer array from stride and kernelShape attribute
    auto strideArgs = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, two32, 0);
    auto kernelShapeArgs = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, two32, 0);
    for (int i = 0; i < 2; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto strideArgsI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          strides[i].cast<IntegerAttr>());
      auto strideGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, strideArgs,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, strideArgsI, strideGep);

      auto kernelShapeArgsI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          kernelShapes[i].cast<IntegerAttr>());
      auto kernelShapeGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, kernelShapeArgs,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, kernelShapeArgsI, kernelShapeGep);
    }

    auto handleStructTy =
      LLVM::LLVMStructType::getOpaque("cudnnContext", context);
    auto handlePtrTy =
      LLVM::LLVMPointerType::get(handleStructTy);

    auto handleLoad = rewriter.create<LLVM::LoadOp>(
        loc, handlePtrTy, cudnnAveragePoolHandle);

    // Call C coded library (../csrc/DNNConvFunc.cpp)
    // TODO: Complete lowering without the library
    auto callAveragePool = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CSRC_AVERAGEPOOL,
        {handleLoad, extractInput, inputDim, extractOutput, outputDim,
        kernelShapeArgs, paddingArgs, strideArgs});

    Value callAveragePoolOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callAveragePool);

    rewriter.replaceOp(op, callAveragePoolOutput);

    return success();
  }
};

void mlir::populateDNNAveragePoolToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle) {
  patterns.insert<DNNAveragePoolOpLowering>(ctx, typeConverter);
  cudnnAveragePoolHandle = handle;
}
