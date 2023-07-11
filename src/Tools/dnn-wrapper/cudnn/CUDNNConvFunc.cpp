#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "cudnnCounter.hpp"

#define DEBUG 0

extern "C"
float* DNNConvFunc(cudnnHandle_t cudnnHandle, float *inData_d, int64_t dimX[4],
    float *filterData_d, int64_t dimw[4],
    float *workspace, int64_t workspaceSize, int64_t algoValue, int64_t groups,
    int64_t pads[2], int64_t strides[2], float *outData_d) {

#if DEBUG
  convCounter();
  printf("\ndimX -> %ld, %ld, %ld, %ld\n", dimX[0] , dimX[1] , dimX[2] ,dimX[3]);
  printf("dimw -> %ld, %ld, %ld, %ld\n\n", dimw[0] , dimw[1] , dimw[2] ,dimw[3]);


  float *X;
  X = (float*) malloc(sizeof(float) * dimX[0] * dimX[1] * dimX[2] * dimX[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimX[0] * dimX[1] * dimX[2] * dimX[3], (cudaMemcpyKind) 2);

  printf("[Convolution] inData_t Addr -> %p, Size -> %ld\n", inData_d, sizeof(float) * dimX[0] * dimX[1] * dimX[2] * dimX[3]);
  printf("[Convolution] inData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  free(X);

  float *f;
  f = (float*) malloc(sizeof(float) * dimw[0] * dimw[1] * dimw[2] * dimw[3]);
  cudaMemcpy(f, filterData_d, sizeof(float) * dimw[0] * dimw[1] * dimw[2] * dimw[3], (cudaMemcpyKind) 2);

  printf("[Convolution] filterData_t Addr -> %p, Size -> %ld\n", filterData_d, sizeof(float) * dimw[0] * dimw[1] * dimw[2] * dimw[3]);
  printf("[Convolution] filterData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
  free(f);
#endif

  cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;

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
  cudnnSetConvolutionGroupCount(convDesc, (int)groups);

  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w);
  cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

  float alpha = 1;
  float beta = 0;
  int error = 5555;
  error = (int)cudnnConvolutionForward(cudnnHandle,
      &alpha,
      inTensorDesc,
      inData_d,
      filterDesc,
      filterData_d,
      convDesc,
      (cudnnConvolutionFwdAlgo_t) algoValue,
      (void*) workspace,
      workspaceSize,
      &beta,
      outTensorDesc,
      outData_d);

#if DEBUG
  printf("[Convolution] conv result -> %d\n", error);

  printf("\npads[0, 1], strides[0, 1] : %ld, %ld, %ld, %ld\n", pads[0], pads[1], strides[0], strides[1]);

  float *y;
  y = (float*) malloc(sizeof(float) * out_n * out_c * out_h * out_w);
  cudaMemcpy(y, outData_d, sizeof(float) * out_n * out_c * out_h * out_w, (cudaMemcpyKind) 2);

  printf("[Convolution] outData_t Addr -> %p, Size -> %ld\n", outData_d, sizeof(float) * out_n * out_c * out_h * out_w);
  printf("[Convolution] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n\n",
      y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  free(y);
#endif

  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyTensorDescriptor(inTensorDesc);
  cudnnDestroyTensorDescriptor(outTensorDesc);
  cudnnDestroyFilterDescriptor(filterDesc);

  return outData_d;
}
