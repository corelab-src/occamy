#include <stdio.h>
#include <cuda.h>

extern "C"
int64_t* CUDAExpandInt64Func (
    int64_t* inData_d, int64_t* dimInput, int64_t inputRank,
    int64_t* outData_d, int64_t* dimOutput,
    int64_t* shapeData_d, int64_t* dimShape) {

  int64_t numElement = 1;
  for (int i=0; i<inputRank; i++) {
    numElement *= dimInput[i];
  }

  cudaMemcpy(outData_d, inData_d, sizeof(int64_t)*numElement, cudaMemcpyDeviceToDevice);

  return outData_d;
}
