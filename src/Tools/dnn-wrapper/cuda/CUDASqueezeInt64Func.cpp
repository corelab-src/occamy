#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>

extern "C"
int64_t* CUDASqueezeInt64Func (
    int64_t *inData_d,  int64_t* dimInput,
    int64_t *outData_d, int64_t* dimOutput,
    int32_t axis, int64_t rank) {

#if 0
  printf("\ndimInput -> %ld, %ld, %ld, %ld\n", dimInput[0] , dimInput[1] , dimInput[2] ,dimInput[3]);

  float *X;
  X = (float*) malloc(sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3], (cudaMemcpyKind) 2);

  printf("[MaxPool] inData_d Addr -> %p, Size -> %d\n", inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  printf("[MaxPool] inData_d -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  free(X);

#endif

  int i;
  int numElem = 1;
  for (i=0; i<rank; i++) {
    numElem *= (int)dimInput[i];
  }

  cudaMemcpy(outData_d, inData_d, sizeof(int64_t)*numElem, cudaMemcpyDeviceToDevice);

#if 0
  float *y;
  y = (float*) malloc(sizeof(float) * dimOutput[0] * dimOutput[1] * dimOutput[2] * dimOutput[3]);
  cudaMemcpy(y, outData_d, sizeof(float) * dimOutput[0] * dimOutput[1] * dimOutput[2] * dimOutput[3], (cudaMemcpyKind) 2);

  printf("[MaxPool] outData_t Addr -> %p, Size -> %d\n", outData_d, sizeof(float) * dimOutput[0] * dimOutput[1] * dimOutput[2] * dimOutput[3]);
  printf("[MaxPool] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  free(y);
#endif

  return outData_d;
}
