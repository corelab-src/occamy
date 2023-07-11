//===----------------------------------------------------------------------===//
// DNN to LLVM: DNNExpandOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class DNNExpandOpLowering : public ConvertToLLVMPattern {
public:
  DNNExpandOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::DNNExpandOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    DNNExpandOp expandOp = dyn_cast<DNNExpandOp>(op);

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

    auto input = expandOp.getX();
    auto output = expandOp.getY();
    auto shapeArg = expandOp.getShape();
    auto inputDimAttr = expandOp.getDimX();
    auto outputDimAttr = expandOp.getDimY();
    auto shapeDimAttr = expandOp.getDimShape();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    auto inputRank = inputDimAttr.size();
    auto outputRank = outputDimAttr.size();
    auto shapeRank = shapeDimAttr.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);
    auto convertedShapeArg = castToLLVMStruct(context, typeConverter, rewriter, loc, shapeArg);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});
    auto extractShape = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedShapeArg,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto inputRankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputRank));

    auto inputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);
    auto shapeDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, four32, 0);


    for (int i = 0; i < 4; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      if((long unsigned)i < inputRank) {
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

      if((long unsigned)i < outputRank) {
        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            outputDimAttr[i].cast<IntegerAttr>());
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      } else {
        auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(1));
        auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
      }

      if((long unsigned)i < shapeRank) {
        auto shapeDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            shapeDimAttr[i].cast<IntegerAttr>());
        auto shapeGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, shapeDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, shapeDimI, shapeGep);
      } else {
        auto shapeDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
            rewriter.getI64IntegerAttr(1));
        auto shapeGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, shapeDim,
            ArrayRef<Value>({offset}));
        rewriter.create<LLVM::StoreOp>(loc, shapeDimI, shapeGep);
      }
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    auto callExpand = core_dnn::DNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, DNNRuntimeAPI::API::CUDA_EXPAND,
        {extractInput, inputDim, inputRankConst, extractOutput, outputDim,
        extractShape, shapeDim});

    Value callExpandOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callExpand);

    rewriter.replaceOp(op, callExpandOutput);

    return success();
  }
};

void mlir::populateDNNExpandToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<DNNExpandOpLowering>(ctx, typeConverter);
}
