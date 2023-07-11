#include <stdio.h>
#include <cuda.h>

#define inputDim 6
__global__ void tensorTranspose6DInt64 (
    int64_t* inData_d, int64_t* dimInput,
    int64_t* outData_d, int64_t* perm, int64_t guard) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard){
    int64_t in_str[inputDim]={0,};
    in_str[0]=1;
    for(int j=1; j<=inputDim; j++){
      in_str[j]=in_str[j-1]*dimInput[inputDim-j];
    }

    int64_t outputShape[inputDim]={0,};
    for(int j=0; j<inputDim; j++){
      outputShape[j]=dimInput[perm[j]];
    }

    int64_t out_str[inputDim]={0,};
    out_str[0]=1;
    for(int j=1; j<inputDim; j++){
      out_str[j]=out_str[j-1]*outputShape[inputDim-j];
    }

    int64_t input_index[inputDim];
    input_index[0]=i/in_str[inputDim-1];
    for(int j=1; j<inputDim; j++){
      input_index[j]=(i%in_str[inputDim-j])/in_str[inputDim-1-j];
    }
    int64_t out_index[inputDim];
    for(int j=0; j<inputDim; j++){
      out_index[j]=input_index[perm[j]];
    }

    int64_t output_index_val=0;
    for(int j=0; j<inputDim; j++){
      output_index_val+=out_index[j]*out_str[inputDim-1-j];
    }
    outData_d[output_index_val] = inData_d[i];
  }
}

extern "C"
int64_t* CUDATranspose6DInt64Func (
    int64_t* inData_d, int64_t* dimInput,
    int64_t*  outData_d,  int64_t* dimOutput,
    int64_t* perm, int64_t rank) {

  int64_t guard = 1;
  for (int i=0; i<rank; i++) {
    guard *= dimOutput[i];
  }
  uint64_t numCTA = (guard+1024-1)/1024;

  int64_t* perm_d;
  cudaMalloc((void**)&perm_d, sizeof(int64_t)*6);
  cudaMemcpy(perm_d, perm, sizeof(int64_t)*6, cudaMemcpyHostToDevice);

  int64_t* inShape_d;
  cudaMalloc((void**)&inShape_d, sizeof(int64_t)*6);
  cudaMemcpy(inShape_d, dimInput, sizeof(int64_t)*6, cudaMemcpyHostToDevice);

  tensorTranspose6DInt64 <<<numCTA, 1024>>> (
      inData_d, inShape_d, outData_d, perm_d, guard);

  return outData_d;
}
