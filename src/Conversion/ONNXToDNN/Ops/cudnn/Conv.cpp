#include <iostream>

//===--------- Start of ONNXConvOpToDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommon.hpp"
#include "src/Conversion/ONNXToDNN/ONNXToDNNCommonCUDA.cuh"
#include "src/Dialect/DNN/DNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace std;

struct ONNXConvOpToDNN : public ConversionPattern {
  ONNXConvOpToDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXConvOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp convOp = dyn_cast<ONNXConvOp>(op);

    auto input = operandAdaptor.getX();
    auto weight = operandAdaptor.getW();
    auto bias = operandAdaptor.getB();
    auto output = convOp.getResult();
    auto group = convOp.getGroup();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto weightMemRef = convertToMemRefType(weight.getType());
    auto biasMemRef = convertToMemRefType(bias.getType());
    auto outputMemRef = convertToMemRefType(output.getType());

    int outputRank = outputMemRef.getShape().size();

    SmallVector<int64_t, 2> pads(2, -1);
    auto padsAttributes = dyn_cast<ONNXConvOp>(op).getPads();
    auto padIt = padsAttributes.value();
    pads[0] = padIt[0].cast<IntegerAttr>().getInt();
    pads[1] = padIt[2].cast<IntegerAttr>().getInt();

    SmallVector<int64_t, 2> strides(2, -1);
    auto stridesAttributes =
      dyn_cast<ONNXConvOp>(op).getStrides();//.dyn_cast_or_null<mlir::DenseElementsAttr>();
    if (!stridesAttributes) {
      return emitError(loc, "Stride: unknown strides");
      strides[0] = 1;
      strides[1] = 1;
    } else {
      auto stridesIt = stridesAttributes.value();
      strides[0] = stridesIt[0].cast<IntegerAttr>().getInt();
      strides[1] = stridesIt[1].cast<IntegerAttr>().getInt();
    }

    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto shape = memRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      memRefType.getElementType().getIntOrFloatBitWidth() / 8;

    int64_t padsArr[2] = {pads[0], pads[1]};
    int64_t stridesArr[2] = {strides[0], strides[1]};
    int64_t dimXArr[4] = {
      inputMemRef.getShape()[0],
      inputMemRef.getShape()[1],
      inputMemRef.getShape()[2],
      inputMemRef.getShape()[3]
    };
    int64_t dimwArr[4] = {
      weightMemRef.getShape()[0],
      weightMemRef.getShape()[1],
      weightMemRef.getShape()[2],
      weightMemRef.getShape()[3]
    };

    int64_t convAlgo = calculateConvAlgo(dimXArr, dimwArr, padsArr, stridesArr, group);
    int64_t workspaceSize = calculateWorkspace(dimXArr, dimwArr, padsArr, stridesArr, convAlgo, group);

    //---------- Making DNNConv Operation ----------//

    //------------ Lowering Pattern ------------//
    auto workspaceConst = rewriter.create<arith::ConstantOp>(loc,
        rewriter.getI64IntegerAttr(workspaceSize));
    SmallVector<int64_t, 1> workspaceShape;
    workspaceShape.push_back(1);
    workspaceShape.push_back(workspaceSize);
    auto workspaceMemRefType = MemRefType::get(workspaceShape, rewriter.getF32Type());
    auto mallocWorkspace = rewriter.create<DNNMallocOp>(loc, workspaceMemRefType, workspaceConst);

    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty, sizeBytes);
    auto mallocConv = rewriter.create<DNNMallocOp>(loc, memRefType, sizeConst);

    // Convolution body
    auto convFwd = rewriter.create<DNNConvForwardOp>(loc, memRefType,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        weight, rewriter.getI64ArrayAttr(weightMemRef.getShape()),
        mallocWorkspace, rewriter.getI64IntegerAttr(workspaceSize),
        rewriter.getI64ArrayAttr(pads),
        rewriter.getI64ArrayAttr(strides),
        rewriter.getI64IntegerAttr(convAlgo),
        rewriter.getI64IntegerAttr(group),
        mallocConv);

    DNNAddOp addOp;
    // Bias Addition
    if(biasMemRef) {
      // Compute broadcasting Dim for B at (A + B) instruction.
      // not compute the whole dimension, just make rankB match with rankA.
      // A represents output
      // B represents bias
      SmallVector<int64_t, 1> broadcastedDimB;
      int biasRank = biasMemRef.getShape().size();

      int dimAIdx = outputRank-1;
      int dimBIdx = biasRank-1;
      assert(outputRank > biasRank && "only support dimA > dimB");

      if(outputRank != biasRank) {
        for (int i=outputRank-1; i>=0; i--) {
          if(dimBIdx == -1) {
            broadcastedDimB.insert(broadcastedDimB.begin(), 1);
          } else {
            int dimAI = outputMemRef.getShape()[dimAIdx];
            int dimBI = biasMemRef.getShape()[dimBIdx];

            if (dimAI == dimBI) {
              broadcastedDimB.insert(broadcastedDimB.begin(), (int64_t)dimAI);
              dimAIdx--;
              dimBIdx--;
            } else {
              broadcastedDimB.insert(broadcastedDimB.begin(), 1);
              dimAIdx--;
            }
          }
        }
      } else {
        for (int i=0; i<outputRank; i++) {
          broadcastedDimB.emplace_back(biasMemRef.getShape()[i]);
        }
      }

      addOp = rewriter.create<DNNAddOp>(loc, memRefType,
          convFwd, rewriter.getI64ArrayAttr(memRefType.getShape()),
          bias, rewriter.getI64ArrayAttr(broadcastedDimB),
          FloatAttr::get(rewriter.getF32Type(), 1.f),
          mallocConv, rewriter.getI64ArrayAttr(memRefType.getShape()));
    }

    // Insert dealloc.
    insertDealloc(mallocConv, loc, rewriter);
    insertDealloc(mallocWorkspace, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, mallocConv, loc, rewriter);
    if (!ret) {
      if(biasMemRef)
        ret = addOp.getResult();
      else
        ret = convFwd.getResult();
    }

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXConvOpToDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXConvOpToDNN>(typeConverter, context);
}
//===---------- End of ONNXConvOpToDNN -----------===//

