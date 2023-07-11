#include <stdio.h>
#include <float.h>
#include <cuda.h>

#define DEBUG 0

__global__ void tensorSoftmax (
    float* inData_d, int64_t* dimInput,
    float* outData_d, int64_t* stride_d,
    int64_t rank, int64_t axis,
    float* maxVal_d, int64_t guard) {
  int64_t i = (int64_t)(blockDim.x * blockIdx.x + threadIdx.x);

  if(i<guard) {
    int64_t currIdx[4] = {0,};

    currIdx[0] = i/stride_d[rank-1];
    for (int64_t j=1; j<rank; j++) {
      currIdx[j] = (i%stride_d[rank-j])/(stride_d[rank-1-j]);
    }

    int64_t headIdx = 0;
    for (int64_t j=0; j<rank; j++) {
      if (j == axis)
        headIdx += 0;
      else
        headIdx += stride_d[rank-1-j]*currIdx[j];
    }

    float reducesum = 0;
    for (int64_t j=0; j<dimInput[axis]; j++) {
      reducesum += expf(inData_d[headIdx + j*stride_d[rank-1-axis]] - *(maxVal_d));
    }

    outData_d[i] = expf(inData_d[i] - *(maxVal_d))/reducesum;

    #if DEBUG
    printf("[%ld] reducesum = %f , target exp val = %f\n(%f, %f, %f, %f) -> (%f, %f, %f, %f)",
        i, reducesum,
        expf(inData_d[i] - *(maxVal_d)),

        (inData_d[headIdx + 0*stride_d[rank-1-axis]] - *(maxVal_d)), 
        (inData_d[headIdx + 1*stride_d[rank-1-axis]] - *(maxVal_d)), 
        (inData_d[headIdx + 2*stride_d[rank-1-axis]] - *(maxVal_d)), 
        (inData_d[headIdx + 3*stride_d[rank-1-axis]] - *(maxVal_d)), 

        expf(inData_d[headIdx + 0*stride_d[rank-1-axis]] - *(maxVal_d)), 
        expf(inData_d[headIdx + 1*stride_d[rank-1-axis]] - *(maxVal_d)), 
        expf(inData_d[headIdx + 2*stride_d[rank-1-axis]] - *(maxVal_d)), 
        expf(inData_d[headIdx + 3*stride_d[rank-1-axis]] - *(maxVal_d))
        );
    #endif
  }
}

__global__ void tensorGetMaxVal(float* inData_d, float* max, int64_t guard){
  int64_t i = (int64_t)(blockDim.x * blockIdx.x + threadIdx.x);

  if(i<guard) {
    if(inData_d[i] > *max) {
      float maxTemp = inData_d[i];
      atomicExch((float*)max, maxTemp);
    }
  }
}


extern "C"
float* CUDASoftmaxFunc (
    float* inData_d, int64_t* dimInput,
    float*  outData_d,  int64_t* dimOutput,
    int64_t axis, int64_t rank) {

  int64_t guard = 1;
  int64_t strideTmp = 1;
  int64_t* stride = (int64_t*)malloc(sizeof(int64_t)*rank);

  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];

    stride[i] = strideTmp;
    strideTmp *= dimInput[rank-1-i];
  }
  uint64_t numCTA = (guard+1024-1)/1024;

  float* maxVal_d;
  float FLOAT_MIN = -FLT_MAX;
  cudaMalloc((void**)&maxVal_d, sizeof(float));
  cudaMemcpy(maxVal_d, &FLOAT_MIN, sizeof(float), cudaMemcpyHostToDevice);

  int64_t* stride_d;
  cudaMalloc((void**)&stride_d, sizeof(int64_t)*rank);
  cudaMemcpy(stride_d, stride, sizeof(int64_t)*rank, cudaMemcpyHostToDevice);

  int64_t* dimInput_d;
  cudaMalloc((void**)&dimInput_d, sizeof(int64_t)*rank);
  cudaMemcpy(dimInput_d, dimInput, sizeof(int64_t)*rank, cudaMemcpyHostToDevice);

  tensorGetMaxVal <<< numCTA, 1024 >>> (
      inData_d, maxVal_d, guard);

  tensorSoftmax <<<numCTA, 1024>>> (
      inData_d, dimInput_d, outData_d, stride_d, rank, axis, maxVal_d, guard);

#if DEBUG
  printf("\ndimInput -> %ld, %ld, %ld, %ld\n", dimInput[0] , dimInput[1] , dimInput[2] ,dimInput[3]);

  float *X;
  X = (float*) malloc(sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3], (cudaMemcpyKind) 2);

  printf("[Softmax] inData_d Addr -> %p, Size -> %ld\n", inData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  printf("[Softmax] inData_d -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  free(X);

  float *Y;
  Y = (float*) malloc(sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  cudaMemcpy(Y, outData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3], (cudaMemcpyKind) 2);

  printf("[Softmax] outData_d Addr -> %p, Size -> %ld\n", outData_d, sizeof(float) * dimInput[0] * dimInput[1] * dimInput[2] * dimInput[3]);
  printf("[Softmax] outData_d -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7]);
  free(X);
#endif

  return outData_d;
}
