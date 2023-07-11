add_core_dnn_library(CDONNXToDNN_CUBLAS
    Gemm.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    CDDNNOps
)
