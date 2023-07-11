#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <stdio.h>

#define DEBUG 0

extern "C"
float* CUBLASMatmul2dFunc(int64_t tensorDim,
    float *inDataA_d, int64_t dimInputA[2],
    float *inDataB_d, int64_t dimInputB[2],
    float *inDataC_d, int64_t dimInputC[2],
    float *outData_d, int64_t dimOutput[2],
    float alpha, float beta,
    int64_t transAAttr, int64_t transBAttr) {

#if DEBUG
  printf("\n[CUBLASMatmul2d] dimInputA -> %ld, %ld\n", dimInputA[0] , dimInputA[1]);
  printf("[CUBLASMatmul2d] dimInputB -> %ld, %ld\n", dimInputB[0] , dimInputB[1]);
  printf("[CUBLASMatmul2d] dimOutput -> %ld, %ld\n", dimOutput[0] , dimOutput[1]);

  float *X;
  X = (float*) malloc(sizeof(float) * dimInputA[0] * dimInputA[1]);
  cudaMemcpy(X, inDataA_d, sizeof(float) * dimInputA[0] * dimInputA[1], (cudaMemcpyKind) 2);

  float *X1;
  X1 = (float*) malloc(sizeof(float) * dimInputB[0] * dimInputB[1]);
  cudaMemcpy(X1, inDataB_d, sizeof(float) * dimInputB[0] * dimInputB[1], (cudaMemcpyKind) 2);

  float *X2;
  X2 = (float*) malloc(sizeof(float) * dimOutput[0] * dimOutput[1]);
  cudaMemcpy(X2, inDataC_d, sizeof(float) * dimOutput[0] * dimOutput[1], (cudaMemcpyKind) 2);

  printf("[CUBLASMatmul2d] inData_dA Addr -> %p, Size -> %lu\n", inDataA_d, sizeof(float) * dimInputA[0] * dimInputA[1]);
  printf("[CUBLASMatmul2d] inData_dA -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);

  printf("[CUBLASMatmul2d] inData_dB Addr -> %p, Size -> %lu\n", inDataB_d, sizeof(float) * dimInputB[0] * dimInputB[1]);
  printf("[CUBLASMatmul2d] inData_dB -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n\n",
      X1[0], X1[1], X1[2], X1[3], X1[4], X1[5], X1[6], X1[7]);

  printf("[CUBLASMatmul2d] inData_dC Addr -> %p, Size -> %lu\n", inDataC_d, sizeof(float) * dimOutput[0] * dimOutput[1]);
  printf("[CUBLASMatmul2d] inData_dC -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n\n",
      X2[0], X2[1], X2[2], X2[3], X2[4], X2[5], X2[6], X2[7]);

  free(X);
  free(X1);
  free(X2);
#endif

  cublasOperation_t transA = transAAttr ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = transBAttr ? CUBLAS_OP_T : CUBLAS_OP_N;

  void* workspace;
  size_t workspaceSize = 1024 * 1024 * 4;
  cudaStream_t stream;
  cublasLtHandle_t ltHandle;

  cublasLtCreate(&ltHandle);
  cudaStreamCreate(&stream);
  cudaMalloc(&workspace, workspaceSize);

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL;
  cublasLtMatrixLayout_t Bdesc = NULL;
  cublasLtMatrixLayout_t Cdesc = NULL;
  cublasLtMatrixLayout_t outdesc = NULL;
  cublasLtMatmulPreference_t preference = NULL;

  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

  cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, dimInputA[0], dimInputA[1], dimInputA[1]);
  cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, (void*)&rowOrder, sizeof(rowOrder));

  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, dimInputB[0], dimInputB[1], dimInputB[1]);
  cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, (void*)&rowOrder, sizeof(rowOrder));

  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, dimOutput[0], dimOutput[1], dimOutput[1]);
  cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, (void*)&rowOrder, sizeof(rowOrder));

  cublasLtMatrixLayoutCreate(&outdesc, CUDA_R_32F, dimOutput[0], dimOutput[1], dimOutput[1]);
  cublasLtMatrixLayoutSetAttribute(outdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, (void*)&rowOrder, sizeof(rowOrder));
#if DEBUG
  printf("%ld, %ld / %ld, %ld / %ld, %ld\n",
      dimInputA[0], dimInputA[1],
      dimInputB[0], dimInputB[1],
      dimOutput[0], dimOutput[1]
      );
#endif

  cublasLtMatmulPreferenceCreate(&preference);
  cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

  cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, outdesc, preference, 1, &heuristicResult, &returnedResults);

  if (returnedResults == 0) {
    printf("CUBLAS_STATUS_NOT_SUPPORTED\n");
  }
#if DEBUG
  else {
    printf("returnedResults : %d\n", returnedResults);
  }
#endif

  int matmulResult = (int)cublasLtMatmul(ltHandle,
      operationDesc,
      &alpha,
      inDataA_d,
      Adesc,
      inDataB_d,
      Bdesc,
      &beta,
      outData_d,
      Cdesc,
      outData_d,
      Cdesc,
      &heuristicResult.algo,
      workspace,
      workspaceSize,
      stream);

#if DEBUG
  printf("\n[CUBLASMatmul2d] matmul result : %d\n" , matmulResult);
  float *y;
  y = (float*) malloc(sizeof(float) * dimOutput[0] * dimOutput[1]);
  cudaMemcpy(y, outData_d, sizeof(float) * dimOutput[0] * dimOutput[1], (cudaMemcpyKind) 2);

  printf("[CUBLASMatmul2d] outData_t Addr -> %p, Size -> %lu\n", outData_d, sizeof(float) * dimOutput[0] * dimOutput[1]);
  printf("[CUBLASMatmul2d] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n\n",
      y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  free(y);
#endif

  // descriptors are no longer needed as all GPU work was already enqueued
  if (preference) cublasLtMatmulPreferenceDestroy(preference);
  if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
  if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
  if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
  if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
  if (stream) cudaStreamDestroy(stream);

  cublasLtDestroy(ltHandle);
  cudaFree(workspace);

  return outData_d;
}
