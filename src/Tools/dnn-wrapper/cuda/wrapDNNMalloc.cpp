#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>
#include "malloccounter.hpp"

#define DEBUG 0

#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H
inline
void checkError(cudaError_t func, size_t size) {
  if (func!= cudaSuccess) {
    printf("[Malloc] GPU error: (size: %ld) %s\n", size, cudaGetErrorString(func));
    exit(-1);
  }
}
#endif

extern "C"
cudaError_t wrapDNNMalloc (void** devPtr, size_t size){

#if DEBUG
  mallocCounter();
#endif

  checkError(cudaMalloc (devPtr, size), size);
  cudaMemset(*devPtr, 0, size);

#if DEBUG
  printf("\n[Malloc] devPtr -> %p\n", *(float**)devPtr);
  printf("[Malloc] malloc size -> %ld\n", size);
#endif

  return (cudaError_t)0;
}
