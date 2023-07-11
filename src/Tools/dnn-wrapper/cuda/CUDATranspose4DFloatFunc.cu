#include <stdio.h>
#include <cuda.h>

#define DEBUG 0

inline
void checkErrorTR4d(cudaError_t func) {
  if (func!= cudaSuccess) {
    printf("[Transpose4D] GPU error: %s\n", cudaGetErrorString(func));
    exit(-1);
  }
}
__global__ void tensorTranspose4DFloat (
    float* inData_d, int64_t* dimInput,
    float* outData_d, int64_t* perm, int64_t guard) {
  int64_t i = (int64_t)(blockDim.x * blockIdx.x + threadIdx.x);

  if(i<guard) {
    int64_t str0 = 1;
    int64_t str1 = dimInput[3];
    int64_t str2 = dimInput[3] * dimInput[2];
    int64_t str3 = dimInput[3] * dimInput[2] * dimInput[1];

    int64_t indices[4] = {(i/str3), ((i%str3)/str2), ((i%str2)/str1), (i%str1)};
    int64_t outputShape[4] = {dimInput[perm[0]],dimInput[perm[1]],
      dimInput[perm[2]],dimInput[perm[3]]};

    int64_t outstr0 = 1;
    int64_t outstr1 = outputShape[3];
    int64_t outstr2 = outputShape[3] * outputShape[2];
    int64_t outstr3 = outputShape[3] * outputShape[2] * outputShape[1];

    outData_d [indices[perm[0]] * outstr3 +
              indices[perm[1]] * outstr2 +
              indices[perm[2]] * outstr1 +
              indices[perm[3]] * outstr0] = inData_d [indices[0] * str3 +
                                                      indices[1] * str2 +
                                                      indices[2] * str1 +
                                                      indices[3] * str0];
  }
}

extern "C"
float* CUDATranspose4DFloatFunc (
    float* inData_d, int64_t* dimInput,
    float*  outData_d,  int64_t* dimOutput,
    int64_t* perm, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  uint64_t numCTA = (guard+1024-1)/1024;

  int64_t* perm_d;
  checkErrorTR4d( cudaMalloc((void**)&perm_d, sizeof(int64_t)*4) );
  checkErrorTR4d( cudaMemcpy(perm_d, perm, sizeof(int64_t)*4, cudaMemcpyHostToDevice) );

  int64_t* inShape_d;
  checkErrorTR4d( cudaMalloc((void**)&inShape_d, sizeof(int64_t)*4) );
  checkErrorTR4d( cudaMemcpy(inShape_d, dimInput, sizeof(int64_t)*4, cudaMemcpyHostToDevice) );

#if DEBUG
  printf("\ndimInput -> %ld, %ld, %ld, %ld\n", dimInput[0] , dimInput[1] , dimInput[2] ,dimInput[3]);

  float *X;
  X = (float*) malloc(sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3], (cudaMemcpyKind) 2);

  printf("[Transpose4DFloat] inData_d Addr -> %p, Size -> %ld\n", inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  printf("[Transpose4DFloat] inData_d -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  free(X);

#endif

  tensorTranspose4DFloat <<<numCTA, 1024>>> (
      inData_d, inShape_d, outData_d, perm_d, guard);

  return outData_d;
}
