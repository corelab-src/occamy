#include <stdio.h>
#include <cuda.h>

__global__ void tensorReciprocal (float* inData_d, float* outData_d, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard) {
    outData_d[i] = 1.f / inData_d[i];
  }

}

extern "C"
float* CUDAReciprocalFunc (
    float* inData_d, float* outData_d,
    int64_t* dimInput, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimInput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;

  tensorReciprocal <<<numCTA, 1024>>> (
      inData_d, outData_d, guard);

  return outData_d;
}
