add_core_dnn_library(CDONNXToDNN_CUDA
    Cast.cpp
    Clip.cpp
    Concat.cpp
    Expand.cpp
    Flatten.cpp
    Unsqueeze.cpp
    UnsqueezeV11.cpp
    Squeeze.cpp
    SqueezeV11.cpp
    Transpose.cpp
    Gather.cpp
    Reshape.cpp
    NonZero.cpp
    Erf.cpp
    Pow.cpp
    MatMul.cpp
    PRelu.cpp
    Pad.cpp
    LeakyRelu.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    CDDNNOps
)
