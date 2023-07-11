#include <stdio.h>
#include <cuda.h>

__global__ void tensorTranspose4DInt64 (
    int64_t* inData_d, int64_t* dimInput,
    int64_t* outData_d, int64_t* perm, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

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
int64_t* CUDATranspose4DInt64Func (
    int64_t* inData_d, int64_t* dimInput,
    int64_t*  outData_d,  int64_t* dimOutput,
    int64_t* perm, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;

  int64_t* perm_d;
  cudaMalloc((void**)&perm_d, sizeof(int64_t)*4);
  cudaMemcpy(perm_d, perm, sizeof(int64_t)*4, cudaMemcpyHostToDevice);

  int64_t* inShape_d;
  cudaMalloc((void**)&inShape_d, sizeof(int64_t)*4);
  cudaMemcpy(inShape_d, dimInput, sizeof(int64_t)*4, cudaMemcpyHostToDevice);

  tensorTranspose4DInt64 <<<numCTA, 1024>>> (
      inData_d, inShape_d, outData_d, perm_d, guard);

  return outData_d;
}
