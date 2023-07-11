add_core_dnn_library(CDONNXToDNN_CUDNN
    Conv.cpp
    MaxPoolSingleOut.cpp
    AveragePool.cpp
    Relu.cpp
    Sigmoid.cpp
    Tanh.cpp
    Reduce.cpp
    ReduceMeanV13.cpp
    Add.cpp
    Sub.cpp
    Mul.cpp
    Div.cpp
    Sqrt.cpp
    Softmax.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    CDONNXToDNN_COMMON_CUDA
    CDDNNOps
)

