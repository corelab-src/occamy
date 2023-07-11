#include <stdio.h>
#include <cuda.h>

#define inputDim 6

__global__ void tensorTranspose6DFloat (
    float* inData_d, int64_t* dimInput,
    float* outData_d, int64_t* perm, int64_t guard) {
  int64_t i = (int64_t)(blockDim.x * blockIdx.x + threadIdx.x);

  if(i<guard) {
    int64_t str0 = 1;
    int64_t str1 = dimInput[5];
    int64_t str2 = dimInput[5] * dimInput[4];
    int64_t str3 = dimInput[5] * dimInput[4] * dimInput[3];
    int64_t str4 = dimInput[5] * dimInput[4] * dimInput[3] * dimInput[2];
    int64_t str5 = dimInput[5] * dimInput[4] * dimInput[3] * dimInput[2] * dimInput[1];
  

    int64_t indices[6] = {(i/str5), ((i%str5)/str4), ((i%str4)/str3), ((i%str3)/str2), ((i%str2)/str1), (i%str1)};
    int64_t outputShape[6] = {
      dimInput[perm[0]],dimInput[perm[1]],
      dimInput[perm[2]],dimInput[perm[3]], 
      dimInput[perm[4]],dimInput[perm[5]]
    };

    int64_t outstr0 = 1;
    int64_t outstr1 = outputShape[5];
    int64_t outstr2 = outputShape[5] * outputShape[4];
    int64_t outstr3 = outputShape[5] * outputShape[4] * outputShape[3];
    int64_t outstr4 = outputShape[5] * outputShape[4] * outputShape[3] * outputShape[2];
    int64_t outstr5 = outputShape[5] * outputShape[4] * outputShape[3] * outputShape[2] * outputShape[1];

    outData_d [indices[perm[0]] * outstr5 +
              indices[perm[1]] * outstr4 +
              indices[perm[2]] * outstr3 +
              indices[perm[3]] * outstr2 +
              indices[perm[4]] * outstr1 +
              indices[perm[5]] * outstr0] = inData_d [indices[0] * str5 +
                                                      indices[1] * str4 +
                                                      indices[2] * str3 +
                                                      indices[3] * str2 +
                                                      indices[4] * str1 +
                                                      indices[5] * str0];
  }
}

extern "C"
float* CUDATranspose6DFloatFunc (
    float* inData_d, int64_t* dimInput,
    float*  outData_d,  int64_t* dimOutput,
    int64_t* perm, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) { guard *= dimOutput[i]; }
  uint64_t numCTA = (guard+1024-1)/1024;

  int64_t* perm_d;
  cudaMalloc((void**)&perm_d, sizeof(int64_t)*6);
  cudaMemcpy(perm_d, perm, sizeof(int64_t)*6, cudaMemcpyHostToDevice);

  int64_t* inShape_d;
  cudaMalloc((void**)&inShape_d, sizeof(int64_t)*6);
  cudaMemcpy(inShape_d, dimInput, sizeof(int64_t)*6, cudaMemcpyHostToDevice);

  tensorTranspose6DFloat <<<numCTA, 1024>>> (
      inData_d, inShape_d, outData_d, perm_d, guard);

  return outData_d;
}
