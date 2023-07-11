#include <stdio.h>
#include <cuda.h>

#define DEBUG 0

//CUDA Kernel for ONNX Gather function (== numpy.take with axis data)
//covers inputRank <= 4, indicesRank <= 4
//which in result covers outputRank <= 7
__global__ void tensorGatherI64 (
    int64_t* X, int64_t* dimInput, int64_t inputDim, 
    int64_t* indices, int64_t* dimIndices, int64_t indicesDim, 
    int64_t axis, int64_t* Y, 
    int guard) {
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<guard){
    int64_t in_str[4]={0,};
    in_str[0]=1;
    for(int j=1; j<inputDim; j++){
      in_str[j]=in_str[j-1]*dimInput[inputDim-j];
    }

    int64_t ind_str[4];
    ind_str[0]=1;
    for(int j=1; j<indicesDim; j++){
      ind_str[j]=ind_str[j-1]*dimIndices[indicesDim-j];
    }
    int64_t ind_size=1;
    for(int j=0; j<indicesDim; j++){
      ind_size*=dimIndices[j];
    }

    int64_t dimOutput[4];
    for(int j=0; j<inputDim; j++){
      dimOutput[j]=dimInput[j];
    }
    dimOutput[axis]=ind_size;

    int64_t out_str[4];
    out_str[0]=1;
    for(int j=1; j<inputDim; j++){
      out_str[j]=out_str[j-1]*dimOutput[inputDim-j];
    }
    int64_t out_index[4];
    out_index[0]=i/out_str[inputDim-1];
    for(int j=1; j<inputDim; j++){
      out_index[j]=(i%out_str[inputDim-j])/out_str[inputDim-1-j];
    }

    int64_t input_index[4];
    for(int j=0; j<inputDim; j++){
      input_index[j]=out_index[j];
    }
    input_index[axis]=indices[out_index[axis]];

    int64_t input_index_val=0;
    for(int j=0; j<inputDim; j++){
      input_index_val+=input_index[j]*in_str[inputDim-1-j];
    }
    Y[i] = X[input_index_val];
  }
}

extern "C"
int64_t* CUDAGatherInt64Func (
    int64_t* inData_d, int64_t* dimInput, int64_t inputRank,
    int64_t* outData_d, int64_t* dimOutput, int64_t outputRank,
    int64_t* indicesData_d, int64_t* dimIndices, int64_t indicesRank,
    int64_t axis) {

  int64_t* dimInput_d;
  cudaMalloc((void**)&dimInput_d, sizeof(int64_t*)*inputRank);
  cudaMemcpy(dimInput_d, dimInput, sizeof(int64_t*)*inputRank, cudaMemcpyHostToDevice);

  int64_t* dimIndices_d;
  cudaMalloc((void**)&dimIndices_d, sizeof(int64_t*)*indicesRank);
  cudaMemcpy(dimIndices_d, dimIndices, sizeof(int64_t*)*indicesRank, cudaMemcpyHostToDevice);

  int64_t guard=1;
  for(int i=0; i<outputRank; i++) guard*=dimOutput[i];
  int64_t numCTA = (guard+1024-1)/1024;

  tensorGatherI64 <<<numCTA, 1024>>> (inData_d, dimInput_d, inputRank, indicesData_d, dimIndices_d, indicesRank, axis, outData_d, guard);

#if DEBUG
  int sizeA = 1;
  for(int i=0; i<inputRank; i++) sizeA*=dimInput[i];

  float *X;
  X = (float*) malloc(sizeof(float) * sizeA);
  cudaMemcpy(X, inData_d, sizeof(float) * sizeA, (cudaMemcpyKind) 2);

  printf("\n[Gather] tensor size : %d", sizeA);

  if(sizeA >= 8)
    printf("\n[Gather] inData_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",
        X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  else
    printf("[Gather] inData_d -> %.5f\n", X[0]);

  float *Z;
  Z = (float*) malloc(sizeof(float) * guard);
  cudaMemcpy(Z, outData_d, sizeof(float) * guard, (cudaMemcpyKind) 2);

  printf("[Gather] outData_d Addr -> %p\n", outData_d);
  printf("[Gather] outData_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n\n\n",
      Z[0], Z[1], Z[2], Z[3], Z[4], Z[5], Z[6], Z[7]);
  free(Z);
#endif


  return outData_d;
}
