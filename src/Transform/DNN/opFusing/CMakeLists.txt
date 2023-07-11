add_core_dnn_library(CDDNNTransformOpFusing
  fuseConvBiasRelu.cpp

  LINK_LIBS PUBLIC
  CDDNNOps
  MLIRTransformUtils
)
