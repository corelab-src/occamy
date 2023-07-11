#include <stdio.h>
#include <cuda.h>

__global__ void tensorLeakyRelu (
    float* inData_d, float* outData_d,
    float alpha, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard) {
    if(inData_d[i] < 0)
      outData_d[i] = alpha * inData_d[i];
    else 
      outData_d[i] = inData_d[i];
  }
}

extern "C"
float* CUDALeakyReluFunc (
    float* inData_d,
    float* outData_d, int64_t* dimOutput,
    float alpha, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;


  tensorLeakyRelu <<<numCTA, 1024>>> (
     inData_d, outData_d, alpha, guard);

  return outData_d;
}
