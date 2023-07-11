add_core_dnn_library(CDDNNToLLVM_CUDA
    Malloc.cpp
    MemPoolInit.cpp
    MemOffset.cpp
    Dealloc.cpp
    Cast.cpp
    Clip.cpp
    Memcpy.cpp
    Concat.cpp
    Reciprocal.cpp
    Negative.cpp
    Erf.cpp
    Flatten.cpp
    Reshape.cpp
    Squeeze.cpp
    Unsqueeze.cpp
    Transpose.cpp
    Expand.cpp
    Gather.cpp
    NonZero.cpp
    Pow.cpp
    MatmulNd.cpp
    PRelu.cpp
    Softmax.cpp
    LeakyRelu.cpp

    LINK_LIBS PUBLIC
    CDDNNRuntime
    CDDNNOps
)

