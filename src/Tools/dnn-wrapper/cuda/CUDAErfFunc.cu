#include <stdio.h>
#include <cuda.h>

__global__ void tensorErf (float* inData_d, float* outData_d, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard) {
    outData_d[i] = erff(inData_d[i]);
  }
}

extern "C"
float* CUDAErfFunc (
    float* inData_d, float*  outData_d,
    int64_t* dimOutput, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;

  tensorErf <<<numCTA, 1024>>> (
     inData_d, outData_d, guard);

  return outData_d;
}
