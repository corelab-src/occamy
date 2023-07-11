#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>

#define DEBUG 0

extern "C"
float* DNNAveragePoolFunc(cudnnHandle_t cudnnHandle, float *inData_d, int64_t dimInput[4], float *outData_d, int64_t dimOutput[4],
                        int64_t kernel[2], int64_t padding[4], int64_t stride[2]) {

#if DEBUG
  printf("\ndimInput -> %ld, %ld, %ld, %ld\n", dimInput[0] , dimInput[1] , dimInput[2] ,dimInput[3]);
  printf("\ndimOutput -> %ld, %ld, %ld, %ld\n\n", dimOutput[0] , dimOutput[1] , dimOutput[2] ,dimOutput[3]);

  printf("kernel, pads, stride : %ld, %ld / %ld, %ld / %ld, %ld\n", kernel[0], kernel[1], padding[0], padding[2], stride[0], stride[1]);

  float *X;
  X = (float*) malloc(sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3], (cudaMemcpyKind) 2);

  printf("[AvgPool] inData_d Addr -> %p, Size -> %d\n", inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  printf("[AvgPool] inData_d -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  free(X);

#endif

  cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;
  cudnnPoolingDescriptor_t poolDesc;

  cudnnCreateTensorDescriptor(&inTensorDesc);
  cudnnCreateTensorDescriptor(&outTensorDesc);
  cudnnCreatePoolingDescriptor(&poolDesc);

  cudnnSetTensor4dDescriptor(
      inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimInput[0], dimInput[1], dimInput[2], dimInput[3]);

  cudnnSetTensor4dDescriptor(
      outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);

  cudnnSetPooling2dDescriptor(poolDesc,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN,
        kernel[0], kernel[1],
        padding[0], padding[2],
        stride[0], stride[1]);

  float alpha = 1;
  float beta = 1;

  cudnnPoolingForward(cudnnHandle,
      poolDesc,
      &alpha,
      inTensorDesc,
      inData_d,
      &beta,
      outTensorDesc,
      outData_d);

#if DEBUG
  float *y;
  y = (float*) malloc(sizeof(float) * dimOutput[0] * dimOutput[1] * dimOutput[2] * dimOutput[3]);
  cudaMemcpy(y, outData_d, sizeof(float) * dimOutput[0] * dimOutput[1] * dimOutput[2] * dimOutput[3], (cudaMemcpyKind) 2);

  printf("[AvgPool] outData_t Addr -> %p, Size -> %d\n", outData_d, sizeof(float) * dimOutput[0] * dimOutput[1] * dimOutput[2] * dimOutput[3]);
  printf("[AvgPool] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  free(y);
#endif

  cudnnDestroyTensorDescriptor(inTensorDesc);
  cudnnDestroyTensorDescriptor(outTensorDesc);
  cudnnDestroyPoolingDescriptor(poolDesc);

  return outData_d;
}
