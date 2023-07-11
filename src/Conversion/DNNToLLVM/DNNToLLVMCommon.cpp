
#include "src/Conversion/DNNToLLVM/DNNRuntimeAPI.hpp"
#include "src/Conversion/DNNToLLVM/DNNToLLVMCommon.hpp"

using namespace mlir;
using namespace core_dnn;

Value mlir::insertAndReturnOutputShapeInfo (
    MLIRContext* context, Location loc, TypeConverter* typeConverter,
    ConversionPatternRewriter &rewriter, Value cudnnOutput, Value llvmOp) {

  auto outputMemRefType = cudnnOutput.getType().cast<mlir::MemRefType>();
  auto elemType = typeConverter->convertType(outputMemRefType.getElementType());
  auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
  auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
  auto outputMemRefShape = outputMemRefType.getShape();
  auto outputRank = outputMemRefShape.size();

  auto int64Ty = IntegerType::get(context, 64);
  auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
  auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, outputRank);

  auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
      rewriter.getI64IntegerAttr(0));
  auto one64 = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
      rewriter.getI64IntegerAttr(1));

  SmallVector<mlir::Type, 3> outputTys(
      {llvmPtrType, llvmPtrType, int64Ty, int64ArrayTy, int64ArrayTy});
  if (outputRank ==0) {
    // Deal with the non ranked value like single int 64 data
    outputTys.erase(outputTys.end()-1);
    outputTys.erase(outputTys.end()-1);
  }

  auto returnTy = LLVM::LLVMStructType::getLiteral(context, outputTys);

  // Shape, stride info of this memref
  LLVM::ConstantOp* shape = (LLVM::ConstantOp*)malloc(sizeof(LLVM::ConstantOp)*outputRank);
  LLVM::ConstantOp* stride = (LLVM::ConstantOp*)malloc(sizeof(LLVM::ConstantOp)*outputRank);
  int64_t st = 1;
  for (int i = outputRank-1; i >= 0; i--) {
    shape[i] = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(outputMemRefShape[i]));
    stride[i] = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(st));
    st *= outputMemRefShape[i];
  }

  auto undef = rewriter.create<LLVM::UndefOp>(loc, returnTy);
  auto insert0 = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, undef, llvmOp,
      llvm::ArrayRef<int64_t>{0});
  // TODO: fix second element
  auto insert1 = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insert0, llvmOp,
      llvm::ArrayRef<int64_t>{1});
  auto insert2 = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insert1, zero64,
      llvm::ArrayRef<int64_t>{2});

  Value insertLast = NULL;
  Value insertTemp = insert2;
  if (outputRank != 0) {
    for (int i=0; i<outputRank; i++) {
      auto insertShape = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insertTemp, shape[i],
          llvm::ArrayRef<int64_t>{3, i});
      auto insertStride = rewriter.create<LLVM::InsertValueOp>(loc, returnTy, insertShape, stride[i],
          llvm::ArrayRef<int64_t>{4, i});
      insertTemp = insertStride;
      if(i==outputRank-1)
        insertLast = insertStride;
    }
  } else {
    // Deal with the non ranked value like single int 64 data
    insertLast = insert2;
  }

  if (!insertLast)
    assert(0 && "Unreachable point: Last insert value is NULL");
  return insertLast;
}

Value mlir::castToLLVMStruct(MLIRContext *context, TypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, Location &loc, Value v){
  if (v.getType().dyn_cast_or_null<mlir::MemRefType>()) {
    auto memRefType = v.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(memRefType.getElementType());
    auto llvmelemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto allocPtrType = LLVM::LLVMPointerType::get(llvmelemType);
    auto int64Ty = IntegerType::get(context, 64);
    int64_t rank;
    // XXX: I do not know how to set type when the memRefType is not ranked.
    if (memRefType.hasRank())
      rank = memRefType.getRank();
    else
      rank = 1;
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, rank);
    SmallVector<mlir::Type, 3> outputTys(
        {allocPtrType, allocPtrType, int64Ty, int64ArrayTy, int64ArrayTy});
    if (memRefType.getShape().size() == 0) {
      // Deal with the non ranked value like single int 64 data
      outputTys.erase(outputTys.end()-1);
      outputTys.erase(outputTys.end()-1);
    }
    auto convertTy = LLVM::LLVMStructType::getLiteral(context, outputTys);

    return rewriter.create<UnrealizedConversionCastOp>(loc,convertTy,v).getResult(0);
  } else return v;
}

void mlir::populateDNNToLLVMConversionPatterns( RewritePatternSet &patterns,
    MLIRContext *ctx, LLVMTypeConverter &typeConverter, Value handle, Value stream) {
////////////////////////////////// Forward Ops Passes /////////////////////////////////
  // ----------------------------------- CUDA ----------------------------------------//
  mlir::populateDNNMallocToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNMemPoolInitToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNMemOffsetToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNDeallocToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNCastToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNClipToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNMemcpyToLLVMConversionPattern(patterns, ctx, typeConverter, stream);
  mlir::populateDNNConcatToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNReciprocalToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNNegativeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNErfToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNFlattenToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNReshapeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNSqueezeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNUnsqueezeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNTransposeToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNExpandToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNGatherToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNNonZeroToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNPowToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNMatmulNdToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNPReluToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNSoftmaxToLLVMConversionPattern(patterns, ctx, typeConverter);
  mlir::populateDNNLeakyReluToLLVMConversionPattern(patterns, ctx, typeConverter);

  // ------------------------------------ DNN --------------------------------------//
  mlir::populateDNNConvForwardToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNActivationForwardToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNAddToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNMulToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNSqrtToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNReduceToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNMaxPoolToLLVMConversionPattern(patterns, ctx, typeConverter, handle);
  mlir::populateDNNAveragePoolToLLVMConversionPattern(patterns, ctx, typeConverter, handle);

  mlir::populateDNNConvBiasActivForwardToLLVMConversionPattern(patterns, ctx, typeConverter, handle);

  // ------------------------------------ cuBLAS --------------------------------------//
  mlir::populateDNNMatmul2dToLLVMConversionPattern(patterns, ctx, typeConverter);
}

void generateDNNHandle(MLIRContext *context, mlir::ModuleOp &m, Value &cudnnHandle) {
  m.walk([&] (Operation* op){
      if(isa<func::FuncOp>(op)){
        func::FuncOp funcOp = dyn_cast<func::FuncOp>(op);
        auto &parentBlock = funcOp.front();
        OpBuilder builder(&funcOp.front().front());
        const auto &apiRegistry = DNNRuntimeAPIRegistry(m, builder, IntegerType::get(context, 32));

        auto handleStructTy =
          LLVM::LLVMStructType::getOpaque("cudnnContext", context);
        auto handlePtrTy =
          LLVM::LLVMPointerType::get(handleStructTy);
        auto handlePtrAddrTy =
          LLVM::LLVMPointerType::get(handlePtrTy);
        auto llvmI32Ty = IntegerType::get(context, 32);

        auto loc = funcOp.front().front().getLoc();

        auto one32 = builder.create<LLVM::ConstantOp>(
            loc, llvmI32Ty, builder.getI32IntegerAttr(1));
        auto handleAlloca = builder.create<LLVM::AllocaOp>(
            loc, handlePtrAddrTy, one32, 0);
        auto handle = DNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            DNNRuntimeAPI::API::CUDNN_CREATE, {handleAlloca});

        auto handleLoad = builder.create<LLVM::LoadOp>(
            loc, handlePtrTy, handleAlloca);
        auto handleDestroy = DNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            DNNRuntimeAPI::API::CUDNN_DESTROY, {handleLoad});
        handleLoad.getOperation()->moveBefore(&parentBlock.back());
        handleDestroy.getDefiningOp()->moveBefore(&parentBlock.back());
        cudnnHandle = handleAlloca;
      }
  });
}

void generateCUDAStream(MLIRContext *context, mlir::ModuleOp &m, Value &cudaStreamValue) {
  m.walk([&] (Operation* op){
      if(isa<func::FuncOp>(op)){
        func::FuncOp funcOp = dyn_cast<func::FuncOp>(op);
        auto &parentBlock = funcOp.front();
        OpBuilder builder(&funcOp.front().front());
        const auto &apiRegistry = DNNRuntimeAPIRegistry(m, builder, IntegerType::get(context, 32));

        auto streamStructTy =
          LLVM::LLVMStructType::getOpaque("CUstream_st", context);
        auto streamPtrTy =
          LLVM::LLVMPointerType::get(streamStructTy);
        auto streamPtrAddrTy =
          LLVM::LLVMPointerType::get(streamPtrTy);
        auto llvmI32Ty = IntegerType::get(context, 32);

        auto loc = funcOp.front().front().getLoc();

        auto one32 = builder.create<LLVM::ConstantOp>(
            loc, llvmI32Ty, builder.getI32IntegerAttr(1));
        auto streamAlloca = builder.create<LLVM::AllocaOp>(
            loc, streamPtrAddrTy, one32, 0);
        auto stream = DNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            DNNRuntimeAPI::API::STREAM_CREATE, {streamAlloca});

        auto streamLoad = builder.create<LLVM::LoadOp>(
            loc, streamPtrTy, streamAlloca);
        auto streamDestroy = DNNRuntimeAPI::callApi(builder, loc, apiRegistry,
            DNNRuntimeAPI::API::STREAM_DESTROY, {streamLoad});
        streamLoad.getOperation()->moveBefore(&parentBlock.back());
        streamDestroy.getDefiningOp()->moveBefore(&parentBlock.back());
        cudaStreamValue = streamAlloca;
      }
  });
}
