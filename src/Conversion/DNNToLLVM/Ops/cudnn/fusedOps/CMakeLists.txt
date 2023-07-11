add_core_dnn_library(CDDNNToLLVM_CUDNN_FUSED
    ConvBiasActiv.cpp

    LINK_LIBS PUBLIC
    CDDNNRuntime
    CDDNNOps
)

