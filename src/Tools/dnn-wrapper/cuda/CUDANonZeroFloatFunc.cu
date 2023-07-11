#include <stdio.h>
#include <cuda.h>

__global__ void tensorNonZeroF32 (
    float* inData_d, float* outData_d, int64_t num) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i == 0) {
    int currIdx = 0;
    for (int j = 0; j < num; j++) {
      if(inData_d[j] != 0.f) {
        outData_d[currIdx] = (float)j;
        currIdx++;
      }
    }
  }
}

extern "C"
float* CUDANonZeroFloatFunc (
    float* inData_d, int64_t* dimInput,
    float*  outData_d, int64_t* dimOutput, int64_t rank) {

  int64_t elemNum = 1;
  for (int64_t i=0; i<rank; i++) {
    elemNum *= dimInput[i];
  }

  tensorNonZeroF32 <<<1, 1>>> (
      inData_d, outData_d, elemNum);

  return outData_d;
}
