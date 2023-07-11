#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>

#define DEBUG 0

extern "C"
float* DNNActiveFunc(cudnnHandle_t cudnnHandle, float *inData_d, int64_t dimX[4], int32_t mode, float* outData_d) {

#if DEBUG
  printf("[Activation] dimX -> %ld, %ld, %ld, %ld\n", dimX[0], dimX[1], dimX[2], dimX[3]);

  float *X;
  X = (float*) malloc(sizeof(float) *  dimX[0]* dimX[1]* dimX[2]* dimX[3]);
  cudaMemcpy(X, inData_d, sizeof(float) *  dimX[0]* dimX[1]* dimX[2]* dimX[3], (cudaMemcpyKind) 2);

  printf("[Activation] inData_t Addr -> %p\n", inData_d);
  printf("[Activation] inData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
#endif


  cudnnTensorDescriptor_t inTensorDesc;

  cudnnCreateTensorDescriptor(&inTensorDesc);

  cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimX[0], dimX[1], dimX[2], dimX[3]);

  float alpha = 1;
  float beta = 0;

  // Describe the activation
  cudnnActivationDescriptor_t activDesc;
  cudnnCreateActivationDescriptor(&activDesc);

  cudnnSetActivationDescriptor(activDesc,
      /* mode -> Values
         CUDNN_ACTIVATION_SIGMOID
         Selects the sigmoid function.

         CUDNN_ACTIVATION_RELU
         Selects the rectified linear function.

         CUDNN_ACTIVATION_TANH
         Selects the hyperbolic tangent function.

         CUDNN_ACTIVATION_CLIPPED_RELU
         Selects the clipped rectified linear function.

         CUDNN_ACTIVATION_ELU
         Selects the exponential linear function.

         CUDNN_ACTIVATION_IDENTITY
         Selects the identity function, intended for bypassing the activation step
         in cudnnConvolutionBiasActivationForward(). (The
         cudnnConvolutionBiasActivationForward() function must use
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM.)
       */
      (cudnnActivationMode_t) mode,
      /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
      /*relu_coef=*/0);

  // Perform the forward pass of the activation
  cudnnActivationForward(cudnnHandle,
      activDesc,
      &alpha,
      inTensorDesc,
      inData_d,
      &beta,
      inTensorDesc,
      outData_d);

#if DEBUG
  cudaMemcpy(X, outData_d, sizeof(float) *  dimX[0]* dimX[1]* dimX[2]* dimX[3], (cudaMemcpyKind) 2);

  int i;
  printf("[Activation] outData_t Addr -> %p\n", outData_d);
  printf("[Activation] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);

  for (i=dimX[0]* dimX[1]* dimX[2]* dimX[3]-6; i<dimX[0]* dimX[1]* dimX[2]* dimX[3];i++) {
    printf("[Activation] result end val -> %.9f\n", X[i]);
  }
#endif

  cudnnDestroyActivationDescriptor(activDesc);
  cudnnDestroyTensorDescriptor(inTensorDesc);

  return outData_d;
}
