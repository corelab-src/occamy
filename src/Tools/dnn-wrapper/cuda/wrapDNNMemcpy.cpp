#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include "counter.hpp"

#define DEBUG 0

inline
void checkErrorMemcpy(cudaError_t func) {
  if (func!= cudaSuccess) {
    printf("[Memcpy] GPU error: %s\n", cudaGetErrorString(func));
    exit(-1);
  }
}

extern "C"
cudaError_t wrapDNNMemcpy (void* dst, const void* src, size_t count, int kind, cudaStream_t stream) {

#if DEBUG
  memcpyCounter();
  if (kind == 2) {
    printf("\n[Memcpy] Dev -> Host (size: %lu)\n", count);
    printf("[Memcpy] DevPtr -> %p\n", src);
    printf("[Memcpy] HostPtr -> %p\n", dst);
  } else {
    printf("\n[Memcpy] Host -> Dev (size: %lu)\n", count);
    printf("[Memcpy] DevPtr -> %p\n", dst);
    printf("[Memcpy] HostPtr -> %p\n", src);
  }
#endif

  checkErrorMemcpy(cudaMemcpyAsync (dst, src, count, (cudaMemcpyKind)kind, stream));

#if DEBUG
  memcpyCounter();
  int i;
  if (kind == 2) {
    float *X = (float*) dst;

    printf("[Memcpy] result -> %.9f, %.9f, %.9f, %.9f ~ %.9f, %.9f, %.9f, %.9f\n",
        X[0], X[1], X[2], X[3],
        X[(count/8)-4], X[(count/8)-3], X[(count/8)-2], X[(count/8)-1]);
  } else {
    float *X;
    X = (float*) malloc(count);
    cudaMemcpy(X, dst, count, (cudaMemcpyKind) 2);
    float *src1 = (float*)src;

    printf("[Memcpy] DevData  -> %.9f, %.9f, %.9f, %.9f ~ %.9f, %.9f, %.9f, %.9f\n",
        X[0], X[1], X[2], X[3],
        X[(count/8)-4], X[(count/8)-3], X[(count/8)-2], X[(count/8)-1]);
    printf("[Memcpy] HostData  -> %.9f, %.9f, %.9f, %.9f ~ %.9f, %.9f, %.9f, %.9f\n",
        src1[0], src1[1], src1[2], src1[3],
        src1[(count/8)-4], src1[(count/8)-3], src1[(count/8)-2], src1[(count/8)-1]);
    free(X);
  }
#endif


  return (cudaError_t)0;
}
