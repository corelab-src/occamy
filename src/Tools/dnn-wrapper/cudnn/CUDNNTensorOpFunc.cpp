#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>

#define DEBUG 0

extern "C"
float* DNNTensorOpFunc(
    cudnnHandle_t cudnnHandle,
    int64_t opMode,
    float *AData_d, int64_t *dimInputA64, int32_t inputARank,
    float *BData_d, int64_t *dimInputB64, int32_t inputBRank,
    float *outData_d, int64_t *dimOutput64, int32_t outputRank,
    float bias) {
  int checkresult;

  int i;

  int64_t sizeA = 1;
  int64_t sizeB = 1;
  int64_t sizeO = 1;

  int* dimInputA = (int*) malloc(sizeof(int)*inputARank);
  int* strideInputA = (int*) malloc(sizeof(int)*inputARank);
  int strideA = 1;
  for (i=0; i<(int)inputARank; i++) {
    sizeA *= dimInputA64[i];
    dimInputA[i] = (int)dimInputA64[i];

    strideInputA[inputARank-i-1] = strideA;
    strideA *= (int)dimInputA64[inputARank-i-1];
  }

  int* dimInputB = (int*) malloc(sizeof(int)*inputBRank);
  int* strideInputB = (int*) malloc(sizeof(int)*inputBRank);
  int strideB = 1;
  for (i=0; i<(int)inputBRank; i++) {
    sizeB *= dimInputB64[i];
    dimInputB[i] = (int)dimInputB64[i];

    strideInputB[inputBRank-i-1] = strideB;
    strideB *= (int)dimInputB64[inputBRank-i-1];
  }

  int* dimOutput = (int*) malloc(sizeof(int)*outputRank);
  int* strideOutput = (int*) malloc(sizeof(int)*outputRank);
  int strideOut = 1;
  for (i=0; i<(int)outputRank; i++) {
    sizeO *= dimOutput64[i];
    dimOutput[i] = (int)dimOutput64[i];

    strideOutput[outputRank-i-1] = strideOut;
    strideOut *= (int)dimOutput64[outputRank-i-1];
  }

#if DEBUG
  printf("\n\n[TensorOp] Arank:%d dimInputA -> %d, %d, %d, %d\n", inputARank,
      dimInputA[0], dimInputA[1], dimInputA[2], dimInputA[3]);
  printf("[TensorOp] Brank:%d dimInputB -> %d, %d, %d, %d\n", inputBRank,
      dimInputB[0], dimInputB[1], dimInputB[2], dimInputB[3]);
  printf("[TensorOp] Orank:%d dimOutput -> %d, %d, %d, %d\n", outputRank,
      dimOutput[0], dimOutput[1], dimOutput[2], dimOutput[3]);

  float *X;
  X = (float*) malloc(sizeof(float) * sizeA);
  cudaMemcpy(X, AData_d, sizeof(float) * sizeA, (cudaMemcpyKind) 2);

  printf("[TensorOp] AData_d Addr -> %p\n", AData_d);
  if(sizeA >= 8)
    printf("[TensorOp] AData_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",
        X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  else
    printf("[TensorOp] AData_d -> %.5f\n", X[0]);

  float *Y;
  Y = (float*) malloc(sizeof(float) * sizeB);
  cudaMemcpy(Y, BData_d, sizeof(float) * sizeB, (cudaMemcpyKind) 2);

  printf("[TensorOp] BData_d Addr -> %p\n", BData_d);
  if(sizeB >= 8)
  printf("[TensorOp] BData_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",
      Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7]);
  else
    printf("[TensorOp] BData_d -> %.5f\n", Y[0]);
#endif

  cudnnTensorDescriptor_t aTensorDesc;
  cudnnCreateTensorDescriptor(&aTensorDesc);
  cudnnSetTensorNdDescriptor(aTensorDesc, CUDNN_DATA_FLOAT,
      (int)inputARank, dimInputA, strideInputA);

  cudnnTensorDescriptor_t bTensorDesc;
  cudnnCreateTensorDescriptor(&bTensorDesc);
  cudnnSetTensorNdDescriptor(bTensorDesc, CUDNN_DATA_FLOAT,
      (int)inputBRank, dimInputB, strideInputB);

  cudnnTensorDescriptor_t outTensorDesc;
  cudnnCreateTensorDescriptor(&outTensorDesc);
  cudnnSetTensorNdDescriptor(outTensorDesc, CUDNN_DATA_FLOAT,
      (int)outputRank, (const int*)dimOutput, strideOutput);

  float alpha1 = 1;
  float alpha2 = bias;
  float beta = 0;

  cudnnOpTensorDescriptor_t opTensorDesc;
  cudnnCreateOpTensorDescriptor(&opTensorDesc);
  /* cudnnOpTensorOp_t enum list
   *
   * CUDNN_OP_TENSOR_ADD
   * CUDNN_OP_TENSOR_MUL
   * CUDNN_OP_TENSOR_MIN
   * CUDNN_OP_TENSOR_MAX
   * CUDNN_OP_TENSOR_SQRT
   * CUDNN_OP_TENSOR_NOT
   */
  cudnnSetOpTensorDescriptor(opTensorDesc,
      (cudnnOpTensorOp_t)opMode, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);

  checkresult = (int)cudnnOpTensor(
      cudnnHandle, opTensorDesc,
      &alpha1, aTensorDesc, AData_d,
      &alpha2, bTensorDesc, BData_d,
      &beta,   outTensorDesc,     outData_d);


#if DEBUG
  printf("\ncheck cudnnOpTensor result : %d\n\n", checkresult);

  float *Z;
  Z = (float*) malloc(sizeof(float) * sizeO);
  cudaMemcpy(Z, outData_d, sizeof(float) * sizeO, (cudaMemcpyKind) 2);

  printf("[TensorOp] outData_d Addr -> %p\n", outData_d);
  if(sizeO >= 8)
  printf("[TensorOp] outData_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n\n\n",
      Z[0], Z[1], Z[2], Z[3], Z[4], Z[5], Z[6], Z[7]);
  else
    printf("[TensorOp] outData_d -> %.5f\n", Z[0]);

#endif

  cudnnDestroyOpTensorDescriptor(opTensorDesc);
  cudnnDestroyTensorDescriptor(aTensorDesc);
  cudnnDestroyTensorDescriptor(bTensorDesc);
  cudnnDestroyTensorDescriptor(outTensorDesc);

  return outData_d;
}
