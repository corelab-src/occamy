add_core_dnn_library(CDDNNToLLVM_CUDNN
    ConvForward.cpp
    ActivationForward.cpp
    Add.cpp
    Mul.cpp
    Sqrt.cpp
    Reduce.cpp
    MaxPool.cpp
    AveragePool.cpp

    LINK_LIBS PUBLIC
    CDDNNRuntime
    CDDNNToLLVM_CUDNN_FUSED
    CDDNNOps
)

add_subdirectory(fusedOps)
