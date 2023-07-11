#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

#define DEBUG 0

inline
void checkdeallocError(cudaError_t func) {
  if (func!= cudaSuccess) {
    printf("[Dealloc] GPU error: %s\n", cudaGetErrorString(func));
    exit(-1);
  }
}

extern "C"
cudaError_t wrapDNNDealloc (void* devPtr){

#if DEBUG
  printf("\n[Dealloc] devPtr -> %p\n", devPtr);
#endif

  checkdeallocError(cudaFree (devPtr));



  return (cudaError_t)0;
}
