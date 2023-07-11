#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

#define DEBUG 0

#ifndef CUDA_ERROR_MemOffsetCHECK_H
#define CUDA_ERROR_MemOffertCHECK_H
inline
void checkErrorMemOffset(cudaError_t func, size_t size) {
  if (func!= cudaSuccess) {
    printf("[MemOffset-cudaMemset] GPU error: (size: %ld) %s\n", size, cudaGetErrorString(func));
    exit(-1);
  }
}
#endif

extern "C"
cudaError_t wrapDNNMemOffset (void** devPtr, void* basePtr, size_t offset, size_t mallocSize){

#if DEBUG
  printf("\n[MemOffset] basePtr -> %p\n", (float*)basePtr);
  printf("[MemOffset] offset -> %ld\n", offset);
#endif

  *devPtr = basePtr+offset;
  checkErrorMemOffset(cudaMemset(*devPtr, 0, mallocSize), mallocSize);

#if DEBUG
  printf("[MemOffset] devPtr -> %p\n", *(float**)devPtr);
#endif

  return (cudaError_t)0;
}
