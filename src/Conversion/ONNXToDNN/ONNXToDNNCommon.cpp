//===------ Common functions for ONNXToDNN Pass ------===//

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"

// Emit constant operation.
Value emitConstantOp(
    OpBuilder &rewriter, Location loc, Type type, double value) {
  Attribute constantAttr;

  TypeSwitch<Type>(type)
    .Case<Float16Type>(
        [&](Type) { constantAttr = rewriter.getF16FloatAttr((float)value);  })
    .Case<Float32Type>(
        [&](Type) { constantAttr = rewriter.getF32FloatAttr((float)value);  })
      .Case<Float64Type>(
          [&](Type) { constantAttr = rewriter.getF64FloatAttr((float)value);  })
        .Case<IntegerType>([&](Type) {
            auto width = type.cast<IntegerType>().getWidth();
            if (width == 1) {
            constantAttr = rewriter.getBoolAttr(value != 0);
            } else {
            constantAttr =
            rewriter.getIntegerAttr(type, APInt(width, (int64_t)value));
            }
            })
  .Case<IndexType>([&](Type) {
      constantAttr = rewriter.getIntegerAttr(type, (int64_t)value);
      })
  .Default([](Type) { llvm_unreachable("unsupported element type");  });
  return rewriter.create<arith::ConstantOp>(loc, constantAttr);
}


// Compute broadcasting Dim for B at (A + B) instruction.
// not compute the whole dimension, just make rankB match with rankA.
void broadcastOnlyDimension (SmallVector<int64_t>* broadcastedDim,
    MemRefType inputAMemRef, MemRefType inputBMemRef) {

  auto inputAShape = inputAMemRef.getShape();
  auto inputBShape = inputBMemRef.getShape();

  int inputARank = inputAShape.size();
  int inputBRank = inputBShape.size();

  int dimAIdx = inputARank-1;
  int dimBIdx = inputBRank-1;
  int finalRank = inputARank >= inputBRank ? inputARank : inputBRank;

  if(inputARank > inputBRank) {
    for (int i=finalRank-1; i>=0; i--) {
      if(dimBIdx == -1) {
        broadcastedDim->insert(broadcastedDim->begin(), 1);
      } else {
        int dimAI = inputAMemRef.getShape()[dimAIdx];
        int dimBI = inputBMemRef.getShape()[dimBIdx];

        if (dimAI == dimBI) {
          broadcastedDim->insert(broadcastedDim->begin(), (int64_t)dimAI);
          dimAIdx--;
          dimBIdx--;
        } else {
          broadcastedDim->insert(broadcastedDim->begin(), 1);
          dimAIdx--;
          if(dimBI == 1) {
            dimBIdx--;
          }
        }
      }
    }
  } else if(inputARank < inputBRank) {
    for (int i=finalRank-1; i>=0; i--) {
      if(dimAIdx == -1) {
        broadcastedDim->insert(broadcastedDim->begin(), 1);
      } else {
        int dimAI = inputAMemRef.getShape()[dimAIdx];
        int dimBI = inputBMemRef.getShape()[dimBIdx];

        if (dimAI == dimBI) {
          broadcastedDim->insert(broadcastedDim->begin(), (int64_t)dimBI);
          dimAIdx--;
          dimBIdx--;
        } else {
          broadcastedDim->insert(broadcastedDim->begin(), 1);
          dimBIdx--;
          if(dimAI == 1) {
            dimAIdx--;
          }
        }
      }
    }
  }
}

/// Get the corresponding MemRefType of a given TensorType/MemRefType.
MemRefType convertToMemRefType(Type type) {
  MemRefType memRefType;
  auto tensorType = type.dyn_cast<TensorType>();
  if (tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    memRefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  } else {
    memRefType = type.dyn_cast<MemRefType>();
  }
  return memRefType;
}

// Determine if current function returns the result value of the
// current op being lowered. If it does then the result value must
// be copied into host memory.
bool checkInsertMemcpy(Operation *currentOp, int resultIndex) {
  auto parentBlock = currentOp->getBlock();

  bool insertMemcpy = false;
  parentBlock->walk([&insertMemcpy, currentOp, resultIndex](func::ReturnOp op) {
    // If there is at least one result to investigate.
    if (currentOp->getNumResults() > 0) {
      auto result = currentOp->getResult(resultIndex);
      for (const auto &operand : op.getOperands())
        if (operand == result)
          insertMemcpy = true;
    }
  });
  return insertMemcpy;
}

// Allocate a MemRef and copy the result of the current op from device
// memory to the MemRef.
Value insertMemcpyToHost(Operation* currentOp, Value result, Location loc,
    PatternRewriter &rewriter, Value operand, int64_t alignment) {
  MemRefType type = convertToMemRefType(result.getType());
  auto memRefShape = type.getShape();
  auto rank = memRefShape.size();
  auto elemType = type.getElementType();
  auto elemSize = elemType.getIntOrFloatBitWidth();

  int64_t size = elemSize / 8;
  for (unsigned int i = 0; i < rank; ++i)
    size *= memRefShape[i];

  memref::AllocOp alloc;
  if (operand) {
    SmallVector<Value, 4> allocOperands;
    for (unsigned int i = 0; i < rank; ++i)
      if (memRefShape[i] < 0) {
        auto dim = rewriter.create<memref::DimOp>(loc, operand, i);
        allocOperands.push_back(dim);
      }
    // Set alignment attribute. Default value is `-1`, which does not set
    // alignment.
    if (alignment >= 0) {
      IntegerAttr constAlignAttr = rewriter.getI64IntegerAttr(alignment);
      alloc =
          rewriter.create<memref::AllocOp>(loc, type, allocOperands, constAlignAttr);
    } else {
      alloc = rewriter.create<memref::AllocOp>(loc, type, allocOperands);
    }
  } else {
    // Set alignment attribute. Default value is `-1`, which does not set
    // alignment.
    if (alignment >= 0) {
      SmallVector<Value, 4> allocOperandsEmpty;
      IntegerAttr constAlignAttr = rewriter.getI64IntegerAttr(alignment);
      alloc = rewriter.create<memref::AllocOp>(
          loc, type, allocOperandsEmpty, constAlignAttr);
    } else {
      alloc = rewriter.create<memref::AllocOp>(loc, type);
    }
  }

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto *parentBlock = alloc.getOperation()->getBlock();
  if (onnx_mlir::hasAllConstantDimensions(type))
    alloc.getOperation()->moveBefore(&parentBlock->front());

  auto int32Ty = rewriter.getIntegerType(32);
  auto int64Ty = rewriter.getIntegerType(64);
  auto memcpySize = emitConstantOp(rewriter, loc, int64Ty,
      size);
  // create memcpy operator
  rewriter.create<DNNMemcpyOp>(loc, int32Ty, alloc, result,
      memcpySize, rewriter.getI32IntegerAttr(2)); // Type = device to host

  // Suppose the op is returned
  int userNum = 0;
  for (Operation *user : currentOp->getUsers()) {
    if (user) userNum++;
  }

  if(userNum > 1) {
    int currentOpIdx = 0;
    func::ReturnOp returnOp;

    SmallVector<int, 1> opIdxVec;

    parentBlock->walk([&returnOp, &currentOp, &currentOpIdx, &opIdxVec](func::ReturnOp op) {
      returnOp = op;
      for (Value returnArg : op->getOperands()) {
        if(returnArg.getDefiningOp() == currentOp)
          opIdxVec.push_back(currentOpIdx);
        currentOpIdx++;
      }
    });

    for (int i=0; i<opIdxVec.size(); i++)
      returnOp.setOperand(opIdxVec[i], alloc.getResult());

    return nullptr;
  } else {
    return alloc;
  }
}

// Insert dnn.dealloc op for the current op. All ops including
// the returning op requires a dealloc, because this is about device
// memory.
Value insertDealloc(Value alloc, Location loc, PatternRewriter &rewriter) {
  auto int32Ty = rewriter.getIntegerType(32);
  DNNDeallocOp dealloc = rewriter.create<DNNDeallocOp>(loc, int32Ty, alloc);
  auto *parentBlock = alloc.getParentBlock();
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return dealloc;
}


