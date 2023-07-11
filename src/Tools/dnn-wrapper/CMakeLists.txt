set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(cd_dnn_wrapper
  STATIC
  # Forward wrappers
  cudnn/CUDNNConvFunc.cpp
  cudnn/CUDNNActiveFunc.cpp
  cudnn/CUDNNTensorOpFunc.cpp
  cudnn/CUDNNReduceFunc.cpp
  cudnn/CUDNNMaxPoolFunc.cpp
  cudnn/CUDNNAveragePoolFunc.cpp

  cudnn/fusedOp/CUDNNConvBiasReluFunc.cpp

  cublas/CUBLASMatmul2dFunc.cu

  cuda/wrapDNNMalloc.cpp
  cuda/wrapDNNMemOffset.cpp
  cuda/wrapDNNMemPoolInit.cpp
  cuda/wrapDNNMemcpy.cpp
  cuda/wrapDNNDealloc.cpp

  cuda/CUDAFlattenFunc.cpp
  cuda/CUDAReshapeFunc.cpp
  cuda/CUDAErfFunc.cu
  cuda/CUDAClipFunc.cu
  cuda/CUDAPowFunc.cu
  cuda/CUDAReciprocalFunc.cu
  cuda/CUDANegativeFunc.cu
  cuda/CUDAExpandInt64Func.cu
  cuda/CUDAMatmulNdFunc.cu
  cuda/CUDAPReluFunc.cu
  cuda/CUDASoftmaxFunc.cu
  cuda/CUDALeakyReluFunc.cu

  cuda/CUDACastI64toFloatFunc.cu

  cuda/CUDATranspose2DFloatFunc.cu
  cuda/CUDATranspose2DInt64Func.cu
  cuda/CUDATranspose3DFloatFunc.cu
  cuda/CUDATranspose3DInt64Func.cu
  cuda/CUDATranspose4DFloatFunc.cu
  cuda/CUDATranspose4DInt64Func.cu
  cuda/CUDATranspose6DFloatFunc.cu
  cuda/CUDATranspose6DInt64Func.cu

  cuda/CUDAUnsqueezeInt64Func.cpp
  cuda/CUDAUnsqueezeFloatFunc.cpp
  cuda/CUDASqueezeInt64Func.cpp
  cuda/CUDASqueezeFloatFunc.cpp

  cuda/CUDAGatherInt64Func.cu
  cuda/CUDAGatherFloatFunc.cu

  cuda/CUDANonZeroInt64Func.cu
  cuda/CUDANonZeroFloatFunc.cu

  cuda/CUDAConcatInt64Func.cu
  cuda/CUDAConcatFloatFunc.cu
  )

target_include_directories(cd_dnn_wrapper
  PRIVATE
  /usr/local/cuda/include
  )

set_property(GLOBAL APPEND PROPERTY CORE_DNN_WRAPPERS cd_dnn_wrapper)
