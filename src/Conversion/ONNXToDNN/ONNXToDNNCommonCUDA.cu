//===------ Common CUDA functions for ONNXToDNN Pass ------===//

#include <cuda.h>
#include <cudnn.h>
#include <iostream>

#include "src/Conversion/ONNXToDNN/ONNXToDNNCommonCUDA.cuh"

int64_t calculateWorkspace(
    int64_t dimX[4], int64_t dimw[4],
    int64_t pads[2], int64_t strides[2],
    int64_t convAlgorithm, int64_t group) {

  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;

  cudnnCreate(&cudnnHandle);
  cudnnCreateTensorDescriptor(&inTensorDesc);
  cudnnCreateTensorDescriptor(&outTensorDesc);
  cudnnCreateFilterDescriptor(&filterDesc);
  cudnnCreateConvolutionDescriptor(&convDesc);

  cudnnSetTensor4dDescriptor(
      inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimX[0], dimX[1], dimX[2], dimX[3]);

  cudnnSetFilter4dDescriptor(
      filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      dimw[0], dimw[1], dimw[2], dimw[3]);


  cudnnSetConvolution2dDescriptor(convDesc,
      (int)pads[0], (int)pads[1],
      (int)strides[0], (int)strides[1],
      1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  cudnnSetConvolutionGroupCount(convDesc, (int)group);

  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w);
  cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

  size_t sizeInBytes = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
      inTensorDesc,
      filterDesc,
      convDesc,
      outTensorDesc,
      (cudnnConvolutionFwdAlgo_t) convAlgorithm,
      &sizeInBytes);

  return sizeInBytes;
}

int64_t calculateConvAlgo(
    int64_t dimX[4], int64_t dimw[4],
    int64_t pads[4], int64_t strides[4], int64_t group) {

  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;

  cudnnCreate(&cudnnHandle);
  cudnnCreateTensorDescriptor(&inTensorDesc);
  cudnnCreateTensorDescriptor(&outTensorDesc);
  cudnnCreateFilterDescriptor(&filterDesc);
  cudnnCreateConvolutionDescriptor(&convDesc);

  cudnnSetTensor4dDescriptor(
      inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimX[0], dimX[1], dimX[2], dimX[3]);

  cudnnSetFilter4dDescriptor(
      filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      dimw[0], dimw[1], dimw[2], dimw[3]);


  cudnnSetConvolution2dDescriptor(convDesc,
      (int)pads[0], (int)pads[1],
      (int)strides[0], (int)strides[1],
      1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  cudnnSetConvolutionGroupCount(convDesc, (int)group);

  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w);
  cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

  int returnedAlgoNum = -1;
  cudnnConvolutionFwdAlgoPerf_t algoPerf[7];

  cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
      inTensorDesc,
      filterDesc,
      convDesc,
      outTensorDesc,
      7,
      &returnedAlgoNum,
      &algoPerf[0]);
  if (((int64_t) algoPerf[0].algo == 0) || ((int64_t) algoPerf[1].algo == 0) ||
      ((int64_t) algoPerf[2].algo == 0) || ((int64_t) algoPerf[3].algo == 0) ||
      ((int64_t) algoPerf[4].algo == 0) || ((int64_t) algoPerf[5].algo == 0) ||
      ((int64_t) algoPerf[6].algo == 0)) {
    return (int64_t) 0;
  } else {
    return (int64_t)algoPerf[0].algo;
  }
}


