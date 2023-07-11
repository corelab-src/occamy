#include <stdio.h>
#include <cuda.h>

__global__ void tensorCastI64toFloat (int64_t* inData_d, float* outData_d, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard) {
    outData_d[i] = (float)inData_d[i];
  }
}

extern "C"
int64_t* CUDACastI64toFloatFunc (
    int64_t* inData_d, float* outData_d, int64_t* dimOutput, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  uint64_t numCTA = (guard+1024-1)/1024;

  tensorCastI64toFloat <<<numCTA, 1024>>> (
     inData_d, outData_d, guard);

  return inData_d;
}
