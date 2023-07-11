#include <stdio.h>
#include <cuda.h>

__global__ void tensorTranspose2DFloat (
    float* inData_d, int64_t* dimInput,
    float* outData_d, int64_t* perm, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard) {
    int64_t str0 = 1;
    int64_t str1 = dimInput[1];

    int64_t indices[2] = {(i/str1), (i%str1)};
    int64_t outputShape[2] = {dimInput[perm[0]],dimInput[perm[1]]};

    int64_t outstr0 = 1;
    int64_t outstr1 = outputShape[1];

    outData_d [indices[perm[0]] * outstr1 +
              indices[perm[1]] * outstr0] = inData_d [indices[0] * str1 +
                                                      indices[1] * str0];
  }
}

extern "C"
float* CUDATranspose2DFloatFunc (
    float* inData_d, int64_t* dimInput,
    float*  outData_d,  int64_t* dimOutput,
    int64_t* perm, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;

  int64_t* perm_d;
  cudaMalloc((void**)&perm_d, sizeof(int64_t)*2);
  cudaMemcpy(perm_d, perm, sizeof(int64_t)*2, cudaMemcpyHostToDevice);

  int64_t* inShape_d;
  cudaMalloc((void**)&inShape_d, sizeof(int64_t)*2);
  cudaMemcpy(inShape_d, dimInput, sizeof(int64_t)*2, cudaMemcpyHostToDevice);

  tensorTranspose2DFloat <<<numCTA, 1024>>> (
      inData_d, inShape_d, outData_d, perm_d, guard);

  return outData_d;
}
