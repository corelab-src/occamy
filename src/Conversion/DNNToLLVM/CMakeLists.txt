add_core_dnn_library(CDDNNRuntime
    DNNRuntimeAPI.cpp
    DNNRuntimeAPI.hpp

    LINK_LIBS PUBLIC
    CDDNNOps
)

add_core_dnn_library(CDDNNToLLVM
    DNNToLLVMCommon.cpp
    ConvertDNNToLLVM.cpp

    LINK_LIBS PUBLIC
    CDDNNRuntime
    CDDNNToLLVM_CUDNN
    CDDNNToLLVM_CUBLAS
    CDDNNToLLVM_CUDA
    CDDNNOps
)

add_subdirectory(Ops)
