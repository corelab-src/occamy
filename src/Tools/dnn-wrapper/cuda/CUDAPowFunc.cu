#include <stdio.h>
#include <cuda.h>

__global__ void tensorPow (float* inData_d, float* exp, float* outData_d, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard) {
    if(exp[0] == 2)
      outData_d[i] = inData_d[i] * inData_d[i];
    else 
      outData_d[i] = powf(inData_d[i], (float)exp[0]);
  }

}

extern "C"
float* CUDAPowFunc (
    float* inData_d, float* exponent, float* outData_d,
    int64_t* dimOutput, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  int64_t numCTA = (guard+1024-1)/1024;


  tensorPow <<<numCTA, 1024>>> (
     inData_d, exponent, outData_d, guard);

  return outData_d;
}
