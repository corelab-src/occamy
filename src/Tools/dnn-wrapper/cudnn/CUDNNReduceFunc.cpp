#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>

#define DEBUG 0

extern "C"
float* DNNReduceFunc(cudnnHandle_t cudnnHandle,
    int32_t tensorDim,
    float *inData_d, int64_t dimInput[4],
    float *outData_d, int64_t dimOutput[4],
    int32_t mode) {

#if DEBUG
  printf("\ndimInput -> %ld, %ld, %ld, %ld\n", dimInput[0] , dimInput[1] , dimInput[2] ,dimInput[3]);
  printf("\ndimOutput -> %ld, %ld, %ld, %ld\n\n", dimOutput[0] , dimOutput[1] , dimOutput[2] ,dimOutput[3]);


  float *X;
  X = (float*) malloc(sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3], (cudaMemcpyKind) 2);

  printf("[Reduce] inData_t Addr -> %p, Size -> %d\n", inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  if (dimInput[0]*dimInput[1]*dimInput[2]*dimInput[3]>=8)
    printf("[Reduce] inData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
        X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  else
    printf("[Reduce] inData_t -> %.9f\n", X[0]);
  free(X);
#endif


  int* intInDim = (int*) malloc(sizeof(int)*tensorDim);
  int* intOutDim = (int*) malloc(sizeof(int)*tensorDim);

  int* strideInput = (int*) malloc(sizeof(int)*tensorDim);
  int* strideOutput = (int*) malloc(sizeof(int)*tensorDim);

  int strideIn = 1;
  int strideOut = 1;

  for (int i = 0; i < tensorDim; i++) {
    intInDim[i] = dimInput[i];
    intOutDim[i] = dimOutput[i];

    strideInput[tensorDim-i-1] = strideIn;
    strideOutput[tensorDim-i-1] = strideOut;

    strideIn *= (int)dimInput[tensorDim-i-1];
    strideOut *= (int)dimOutput[tensorDim-i-1];
  }

  cudnnReduceTensorOp_t reduceMode = (cudnnReduceTensorOp_t) mode;
  cudnnTensorDescriptor_t inTensorDesc, outTensorDesc;
  cudnnReduceTensorDescriptor_t reduceDesc;

  cudnnCreateTensorDescriptor(&inTensorDesc);
  cudnnCreateTensorDescriptor(&outTensorDesc);
  cudnnCreateReduceTensorDescriptor(&reduceDesc);

  if (tensorDim == 4) {
    cudnnSetTensor4dDescriptor(
        inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        dimInput[0], dimInput[1], dimInput[2], dimInput[3]);
    cudnnSetTensor4dDescriptor(
        outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);

  } else {
    cudnnSetTensorNdDescriptor(
        inTensorDesc, CUDNN_DATA_FLOAT, (int)tensorDim,
        (const int*)intInDim, (const int*)strideInput);
    cudnnSetTensorNdDescriptor(
        outTensorDesc, CUDNN_DATA_FLOAT, (int)tensorDim,
        (const int*)intOutDim, (const int*)strideOutput);
  }

  cudnnSetReduceTensorDescriptor(reduceDesc,
      reduceMode,
      CUDNN_DATA_FLOAT,
      CUDNN_PROPAGATE_NAN,
      CUDNN_REDUCE_TENSOR_NO_INDICES, // Indices generation is only supported
                                      // at the case of REDUCE_MIN or MAX
      CUDNN_32BIT_INDICES);

  size_t idxByteSize = 0;
  cudnnGetReductionIndicesSize(cudnnHandle,
      reduceDesc,
      inTensorDesc,
      outTensorDesc,
      &idxByteSize);

  void * idxSpace;
  if (idxByteSize != 0) {
    cudaMalloc(&idxSpace, idxByteSize);
  }

  size_t workspaceByteSize = 0;
  cudnnGetReductionWorkspaceSize(cudnnHandle,
      reduceDesc,
      inTensorDesc,
      outTensorDesc,
      &workspaceByteSize);

  void * workSpace;
  if (workspaceByteSize != 0) {
    cudaMalloc(&workSpace, workspaceByteSize);
  }

  float alpha = 1.f;
  float beta = 0.f;

  int error = 5837;
  error = (int)cudnnReduceTensor(cudnnHandle,
      reduceDesc,
      idxSpace, idxByteSize,
      workSpace, workspaceByteSize,
      &alpha, inTensorDesc, inData_d,
      &beta, outTensorDesc, outData_d);

#if DEBUG
  printf("[Reduce] Reduce Result value -> %d\n", error);

  float *y;
  y = (float*) malloc(sizeof(float) * dimOutput[0] * dimOutput[1]* dimOutput[2]* dimOutput[3]);
  cudaMemcpy(y, outData_d, sizeof(float) * dimOutput[0] * dimOutput[1]* dimOutput[2]* dimOutput[3], (cudaMemcpyKind) 2);

  printf("[Reduce] outData_t Addr -> %p, Size -> %d\n", outData_d, sizeof(float) * dimOutput[0] * dimOutput[1]* dimOutput[2]* dimOutput[3]);
  if (dimOutput[0]*dimOutput[1]*dimOutput[2]*dimOutput[3]>=8)
    printf("[Reduce] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  else
    printf("[Reduce] outData_t -> %.9f\n", y[0]);
  free(y);
#endif

  cudnnDestroyTensorDescriptor(inTensorDesc);
  cudnnDestroyTensorDescriptor(outTensorDesc);
  cudnnDestroyReduceTensorDescriptor(reduceDesc);

  return outData_d;
}
