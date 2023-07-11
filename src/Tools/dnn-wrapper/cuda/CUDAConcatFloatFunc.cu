#include <stdio.h>
#include <cuda.h>

#define MaxConcat 10
#define MaxRank 4

__global__ void tensorConcatFloat (
    float** Input, int64_t** dimInput, int64_t numInput, int64_t inputRank, 
    float* output, int64_t* dimOutput, int64_t axis,
    int guard) {
  int32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  // guard, prevent problem from large i
  if(i<guard){
    // Calculate stride value for output
    int32_t out_str[MaxRank]={0,};
    out_str[0]=1;
    for(int j=1; j<inputRank; j++){
      out_str[j]=out_str[j-1]*dimOutput[inputRank-j];
    }

    // Calaulate output index with i
    int32_t out_index[4];
    out_index[0]=i/out_str[inputRank-1];
    for(int j=1; j<inputRank; j++){
      out_index[j]=(i%out_str[inputRank-j])/out_str[inputRank-1-j];
    }
    // Calculate accumulated value for input
    int32_t in_acc[MaxConcat+1]={0,};
    in_acc[0]=0;
    for(int j=0; j<numInput; j++){
      in_acc[j+1]=in_acc[j]+dimInput[j][axis];
    }

    // Calculate source input to copy from
    int32_t source_offset=0;
    for(int j=0; j<numInput; j++){
      if(out_index[axis]>=in_acc[j] && out_index[axis]<in_acc[j+1]){
        source_offset=j;
        break;
      }
    }
    // Calculate stride value for input A
    int32_t in_str[MaxRank]={0,};
    in_str[0]=1;
    for(int j=1; j<inputRank; j++){
      in_str[j]=in_str[j-1]*dimInput[source_offset][inputRank-j];
    }

    // Copy output index to input index 
    int32_t input_index[4];
    for(int j=0; j<inputRank; j++){
      input_index[inputRank-1-j]=out_index[inputRank-1-j];
    }

    input_index[axis]=out_index[axis]-in_acc[source_offset];
    int32_t input_index_offset=0;
    for(int j=0; j<inputRank; j++){
      input_index_offset+=input_index[j]*in_str[inputRank-1-j];
    }
    output[i] = Input[source_offset][input_index_offset];
  }
}

extern "C"
float* CUDAConcatFloatFunc (
    float** inDataList, int64_t** dimInputList,
    float*  outData_d,  int64_t* dimOutput,
    int64_t axis, int64_t inputNum, int32_t inputRank) {

  int64_t guard = 1;
  for (int32_t i=0; i<inputRank; i++) { guard *= dimOutput[i]; }
  uint64_t numCTA = (guard+1024-1)/1024;

  float** inDataList_d = NULL;
  cudaMalloc((void**)&inDataList_d, sizeof(float*)*inputNum);
  cudaMemcpy(inDataList_d, inDataList, sizeof(float*)*inputNum, cudaMemcpyHostToDevice);

  int64_t** dimInputList_temp= (int64_t**) malloc(sizeof(int64_t*)*inputNum);
  for (int32_t i=0; i<inputNum; i++) {
    cudaMalloc((void**)&dimInputList_temp[i], sizeof(int64_t)*inputRank);
    cudaMemcpy(dimInputList_temp[i], dimInputList[i],
        sizeof(int64_t)*inputRank, cudaMemcpyHostToDevice);
  }

  int64_t** dimInputList_d = NULL;
  cudaMalloc((void**)&dimInputList_d, sizeof(int64_t*)*inputNum);
  cudaMemcpy(dimInputList_d, dimInputList_temp,
      sizeof(int64_t*)*inputNum, cudaMemcpyHostToDevice);

  int64_t* dimOutput_d;
  cudaMalloc((void**)&dimOutput_d, sizeof(int64_t)*inputRank);
  cudaMemcpy(dimOutput_d, dimOutput, sizeof(int64_t)*inputRank, cudaMemcpyHostToDevice);

  tensorConcatFloat <<<numCTA, 1024>>> (
      inDataList_d, dimInputList_d, inputNum, inputRank,
      outData_d, dimOutput_d, axis, guard);
  
  return outData_d;
}
