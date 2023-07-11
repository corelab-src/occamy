#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

#define DEBUG 0

#ifndef CUDA_ERROR_MEMPOOLINIT_CHECK_H
#define CUDA_ERROR_MEMPOOLINIT_CHECK_H
inline
void checkErrorMemPoolInit(cudaError_t func, size_t size) {
  if (func!= cudaSuccess) {
    printf("[MemPoolInit] GPU error: (size: %ld) %s\n", size, cudaGetErrorString(func));
    exit(-1);
  }
}
#endif

extern "C"
cudaError_t wrapDNNMemPoolInit (void** devPtr, size_t size){

  checkErrorMemPoolInit(cudaMalloc (devPtr, size), size);

#if DEBUG
  printf("\n[MemPoolInit] devPtr -> %p\n", *(float**)devPtr);
  printf("[MemPoolInit] malloc size -> %ld\n", size);
#endif

  return (cudaError_t)0;
}
