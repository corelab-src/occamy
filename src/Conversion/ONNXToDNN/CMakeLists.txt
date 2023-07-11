# SPDX-License-Identifier: Apache-2.0
add_core_dnn_library(CDONNXToDNN_COMMON_CUDA
    ONNXToDNNCommonCUDA.cu

    LINK_LIBS PUBLIC
    cudart
    cudnn
    )

target_include_directories(CDONNXToDNN_COMMON_CUDA
    PRIVATE
    /usr/local/cuda/include
    )


add_core_dnn_library(CDONNXToDNN
    ConvertONNXToDNN.cpp
    ONNXToDNNCommon.cpp
    Ops/Constant.cpp
    Ops/ConstantOfShape.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    CDDNNOps
    CDONNXToDNN_COMMON_CUDA
    CDONNXToDNN_CUDNN
    CDONNXToDNN_CUBLAS
    CDONNXToDNN_CUDA
    )

add_subdirectory(Ops)

