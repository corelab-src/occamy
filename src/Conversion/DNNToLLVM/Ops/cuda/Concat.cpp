//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNConcatOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNConcatOpLowering : public ConvertToLLVMPattern {
public:
  DNNConcatOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNConcatOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNConcatOp concatOp = dyn_cast<DNNConcatOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = DNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64PtrPtrTy = LLVM::LLVMPointerType::get(int64PtrTy);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);
    auto floatPtrPtrTy = LLVM::LLVMPointerType::get(floatPtrTy);

    auto input = concatOp.getX();
    auto output = concatOp.getY();
    auto outputDimAttr = concatOp.getDimY();
    auto axis = concatOp.getAxis();

    auto inputMemRefType = input[0].getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto llvmPtrPtrType = LLVM::LLVMPointerType::get(llvmPtrType);
    auto inputMemRefShape = inputMemRefType.getShape();

    int inputNum = input.size();
    int inputRank = inputMemRefShape.size();

    auto axisConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(axis));
    auto inNumConst = rewriter.create<LLVM::ConstantOp>(loc,
        int64Ty, rewriter.getI64IntegerAttr(inputNum));

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and output
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create inputs array according to the type of the input elements.
    Value inputsArgs;
    auto inputNumConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(inputNum));

    if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
      inputsArgs = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrPtrTy, inputNumConst, 0);

      for (int i = 0; i < inputNum; i++) {
        auto inputOffset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

        // Insert unrealized conversion cast op to convert memref to llvm struct type.
        auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input[i]);

        auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
            llvm::ArrayRef<int64_t>{0});
        auto inputArgsGep = rewriter.create<LLVM::GEPOp>(loc, llvmPtrPtrType, inputsArgs,
            ArrayRef<Value>({inputOffset}));
        rewriter.create<LLVM::StoreOp>(loc, extractInput, inputArgsGep);
      }
    } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
      inputsArgs = rewriter.create<LLVM::AllocaOp>(
        loc, floatPtrPtrTy, inputNumConst, 0);

      for (int i = 0; i < inputNum; i++) {
        auto inputOffset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));
        //
        // Insert unrealized conversion cast op to convert memref to llvm struct type.
        auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input[i]);

        auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
            llvm::ArrayRef<int64_t>{0});
        auto inputArgsGep = rewriter.create<LLVM::GEPOp>(loc, llvmPtrPtrType, inputsArgs,
            ArrayRef<Value>({inputOffset}));
        rewriter.create<LLVM::StoreOp>(loc, extractInput, inputArgsGep);
      }
    } else {
      assert(0 && "not supported elem type");
    }

    // Create integer array from output shape attribute
    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
        rewriter.getI32IntegerAttr(inputRank));
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < inputRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputDimAttr[i].cast<IntegerAttr>());
      auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
    }

    // Create integer array array (int64_t**) from input shape attribute
    rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    auto inputDimsArgs = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrPtrTy, inputNumConst, 0);

    for (int i = 0; i < inputNum; i++) {
      auto dimOffset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto inputMemref = input[i].getType().cast<mlir::MemRefType>();
      auto inputShape = inputMemref.getShape();

      auto inputDim = rewriter.create<LLVM::AllocaOp>(
          loc, int64PtrTy, rankConst, 0);
      for (int i=0; i<inputRank; i++) {
        auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
            rewriter.getI32IntegerAttr(i));

        auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(inputShape[i]));
        auto inputDimGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputDimGep);
      }
      auto inputDimsArgsGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrPtrTy, inputDimsArgs,
          ArrayRef<Value>({dimOffset}));
      rewriter.create<LLVM::StoreOp>(loc, inputDim, inputDimsArgsGep);
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    Value callConcat;
    if(llvmElemType.dyn_cast_or_null<IntegerType>()) {

      callConcat = DNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, DNNRuntimeAPI:: API::CUDA_CONCAT_I64,
          {inputsArgs, inputDimsArgs, extractOutput, outputDim,
          axisConst, inNumConst, rankConst});

    } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {

      callConcat = DNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, DNNRuntimeAPI:: API::CUDA_CONCAT_F32,
          {inputsArgs, inputDimsArgs, extractOutput, outputDim,
          axisConst, inNumConst, rankConst});
    }

    Value callConcatOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callConcat);

    rewriter.replaceOp(op, callConcatOutput);

    return success();
  }
};

void mlir::populateDNNConcatToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNConcatOpLowering>(ctx, typeConverter);
}
