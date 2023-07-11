#include <stdio.h>
#include <cuda.h>

#define DEBUG 0

//CUDA Kernel for Matrix Multiply
//two input tensor and one output tensor
// with corresponding dimension info(Shape of tensor : (1,2,5,5) ) and rank(dimension of tensor : 1d, 2d, 3d, 4d)
// 
// Tensor A : dimA - ((*,) A, B) 
// Tensor B : dimB - ((*,) B, C)
// Tensor output : dimOutput - ((*,), A, C)
// rank of two input tensor may different
__global__ void tensorMatmulNd (
    float* A, int64_t* dimInputA, int32_t inputRankA, 
    float* B, int64_t* dimInputB, int32_t inputRankB, 
    float* output, int64_t* dimOutput, int32_t outputRank, 
    int guard) {

  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i<guard){
    // Calculate stride value for input A
    int32_t in_strA[4]={0,};
    in_strA[0]=1;
    for(int j=1; j<inputRankA; j++){
      in_strA[j]=in_strA[j-1]*dimInputA[inputRankA-j];
    }
    // Calculate stride value for input B
    int32_t in_strB[4]={0,};
    in_strB[0]=1;
    for(int j=1; j<inputRankB; j++){
      in_strB[j]=in_strB[j-1]*dimInputB[inputRankB-j];
    }
    // Calculate stride value for output B
    int32_t out_str[4]={0,};
    out_str[0]=1;
    for(int j=1; j<outputRank; j++){
      out_str[j]=out_str[j-1]*dimOutput[outputRank-j];
    }
    // Calaulate output index with i
    int32_t out_index[4];
    out_index[0]=i/out_str[outputRank-1];
    for(int j=1; j<outputRank; j++){
      out_index[j]=(i%out_str[outputRank-j])/out_str[outputRank-1-j];
    }
    // Copy output index to index A 
    int32_t A_index[4];
    for(int j=0; j<inputRankA; j++){
      A_index[inputRankA-1-j]=out_index[outputRank-1-j];
    }
    // Copy output index to index B 
    int32_t B_index[4];
    for(int j=0; j<inputRankB; j++){
      B_index[inputRankB-1-j]=out_index[outputRank-1-j];
    }

    // Calculate base index of tensor A 
    // set last index of tensor A to 0
    // since matmul iteratively access ...[A_index[inputRankA-2]][j]
    A_index[inputRankA-1]=0;
    int32_t A_index_base=0;
    for(int j=0; j<inputRankA; j++){
      A_index_base+=A_index[j]*in_strA[inputRankA-1-j];
    }
    // Calculate base index of tensor B
    // set second last index of tensor B to 0
    // since matmul iteratively access ...[j][B_index[inputRank-1]]
    B_index[inputRankB-2]=0;
    int32_t B_index_base=0;
    for(int j=0; j<inputRankB; j++){
      B_index_base+=B_index[j]*in_strB[inputRankB-1-j];
    }
    // Calculate output result 
    // multiply and accumulate
    // multiply ...[A_index[inputRankA-2]][j] and ...[j][B_index[inputRank-1]]
    float temp=0;
    for(int j=0; j<dimInputA[inputRankA-1]/*dimInputB[inputRankB-2]*/; j++){
      temp += A[A_index_base+j*in_strA[0]]*B[B_index_base+j*in_strB[1]];
    }
    output[i] = temp;
  }
}

extern "C"
float* CUDAMatmulNdFunc (
    float* inDataA_d, int64_t* dimInputA, int32_t inputRankA,
    float* inDataB_d, int64_t* dimInputB, int32_t inputRankB,
    float* outData_d, int64_t* dimOutput, int32_t outputRank) {

#if DEBUG
  printf("\n[MatmulNd] inDataA_d -> %p\n", inDataA_d);
  printf("inDataA dim  = ");
  for (int i=0; i<inputRankA; i++) {
    printf( "%ld ", dimInputA[i]);
  }
  printf("\n");

  printf("[MatmulNd] inDataB_d -> %p\n", inDataB_d);
  printf("inDataB dim  = ");
  for (int i=0; i<inputRankB; i++) {
    printf( "%ld ", dimInputB[i]);
  }
  printf("\n");

  printf("[MatmulNd] outData_d -> %p\n", outData_d);
  printf("outData dim  = ");
  for (int i=0; i<outputRank; i++) {
    printf( "%ld ", dimOutput[i]);
  }
  printf("\n\n");
#endif

  int64_t* dimInputA_d;
  cudaMalloc((void**)&dimInputA_d, sizeof(int64_t*)*inputRankA);
  cudaMemcpy(dimInputA_d, dimInputA, sizeof(int64_t*)*inputRankA, cudaMemcpyHostToDevice);

  int64_t* dimInputB_d;
  cudaMalloc((void**)&dimInputB_d, sizeof(int64_t*)*inputRankB);
  cudaMemcpy(dimInputB_d, dimInputB, sizeof(int64_t*)*inputRankB, cudaMemcpyHostToDevice);

  int64_t* dimOutput_d;
  cudaMalloc((void**)&dimOutput_d, sizeof(int64_t*)*outputRank);
  cudaMemcpy(dimOutput_d, dimOutput, sizeof(int64_t*)*outputRank, cudaMemcpyHostToDevice);

  int64_t guard=1;
  for(int i=0; i<outputRank; i++) guard*=dimOutput[i];
  int64_t numCTA = (guard+1024-1)/1024;

  tensorMatmulNd <<<numCTA, 1024>>> (
      inDataA_d, dimInputA_d, inputRankA,
      inDataB_d, dimInputB_d, inputRankB,
      outData_d, dimOutput_d, outputRank,
      guard);


#if DEBUG
  int sizeA = 1;
  int sizeB = 1;
  for(int i=0; i<inputRankA; i++) sizeA*=dimInputA[i];
  for(int i=0; i<inputRankB; i++) sizeB*=dimInputB[i];

  float *X;
  X = (float*) malloc(sizeof(float) * sizeA);
  cudaMemcpy(X, inDataA_d, sizeof(float) * sizeA, (cudaMemcpyKind) 2);

  if(sizeA >= 8)
    printf("\n[MatmulNd] inDataA_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",
        X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  else
    printf("[MatmulNd] inDataA_d -> %.5f\n", X[0]);

  float *Y;
  Y = (float*) malloc(sizeof(float) * sizeB);
  cudaMemcpy(Y, inDataB_d, sizeof(float) * sizeB, (cudaMemcpyKind) 2);

  if(sizeB >= 8)
    printf("[MatmulNd] inDataB_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",
        Y[0], Y[1], Y[2], Y[3], Y[4], Y[5], Y[6], Y[7]);
  else
    printf("[MatmulNd] inDataB_d -> %.5f\n", Y[0]);
  float *Z;
  Z = (float*) malloc(sizeof(float) * guard);
  cudaMemcpy(Z, outData_d, sizeof(float) * guard, (cudaMemcpyKind) 2);

  printf("[MatmulNd] outData_d Addr -> %p\n", outData_d);
  printf("[MatmulNd] outData_d -> %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n\n\n",
      Z[0], Z[1], Z[2], Z[3], Z[4], Z[5], Z[6], Z[7]);
  free(Z);
#endif

  return outData_d;
}
