//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNGatherOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNGatherOpLowering : public ConvertToLLVMPattern {
public:
  DNNGatherOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNGatherOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNGatherOp gatherOp = dyn_cast<DNNGatherOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = DNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);

    auto input = gatherOp.getX();
    auto output = gatherOp.getY();
    auto indices = gatherOp.getIndices();
    auto axis = gatherOp.getAxis();

    auto inputDimAttr = gatherOp.getDimX();
    auto outputDimAttr = gatherOp.getDimY();
    auto indicesDimAttr = gatherOp.getDimIndices();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    SmallVector<mlir::Type, 4> outputTys(
        {llvmPtrType, llvmPtrType, int64Ty, int64ArrayTy, int64ArrayTy});

    auto inputRank = inputDimAttr.size();
    auto outputRank = outputDimAttr.size();
    auto indicesRank = indicesDimAttr.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);
    auto convertedIndices = castToLLVMStruct(context, typeConverter, rewriter, loc, indices);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});
    auto extractIndices = rewriter.create<LLVM::ExtractValueOp>(loc, int64PtrTy, convertedIndices, //indicesElemPtrType, indices,
        llvm::ArrayRef<int64_t>{0});

    auto axisConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(axis));

    // Create integer array from shape attribute
    auto inputRankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputRank));
    auto outputRankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(outputRank));
    auto indicesRankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(indicesRank));

    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, inputRankConst, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, outputRankConst, 0);
    auto indicesDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, indicesRankConst, 0);

    for (int i = 0; i < 4; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      if((long unsigned)i < inputRank) {
        auto inputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            inputDimAttr[i].cast<IntegerAttr>());
        auto inputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, inputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, inputDimI, inputGep);
      }

      if((long unsigned)i < outputRank) {
        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            outputDimAttr[i].cast<IntegerAttr>());
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      }

      if((long unsigned)i < indicesRank) {
        auto indicesDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            indicesDimAttr[i].cast<IntegerAttr>());
        auto indicesGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, indicesDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, indicesDimI, indicesGep);
      }
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    Value callGather;
    if(llvmElemType.dyn_cast_or_null<IntegerType>()) {
      callGather = DNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, DNNRuntimeAPI::API::CUDA_GATHER_I64,
          {extractInput, inputDim, inputRankConst,
          extractOutput, outputDim, outputRankConst,
          extractIndices, indicesDim, indicesRankConst,
          axisConst});
    } else if(llvmElemType.dyn_cast_or_null<FloatType>()) {
      callGather = DNNRuntimeAPI::callApi(rewriter, loc,
          apiRegistry, DNNRuntimeAPI::API::CUDA_GATHER_F32,
          {extractInput, inputDim, inputRankConst,
          extractOutput, outputDim, outputRankConst,
          extractIndices, indicesDim, indicesRankConst,
          axisConst});
    }

    Value callGatherOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callGather);

    rewriter.replaceOp(op, callGatherOutput);

    return success();
  }
};

void mlir::populateDNNGatherToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNGatherOpLowering>(ctx, typeConverter);
}
