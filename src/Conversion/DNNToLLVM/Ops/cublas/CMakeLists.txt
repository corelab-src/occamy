add_core_dnn_library(CDDNNToLLVM_CUBLAS
    Matmul2d.cpp

    LINK_LIBS PUBLIC
    CDDNNRuntime
    CDDNNOps
)

