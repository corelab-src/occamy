#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tensorClip(float* input, float* output,
    float minVal, float maxVal, int guard) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < guard) {
    output[i] = fminf(fmaxf(input[i], minVal), maxVal);
  }
}

extern "C"
float* CUDAClipFunc (
    float* inData_d, float*  outData_d,
    int64_t* dimOutput, float min, float max, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;

  tensorClip <<<numCTA, 1024>>> (
     inData_d, outData_d, min, max, guard);

  return outData_d;
}
