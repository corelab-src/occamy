add_core_dnn_library(CDDNNTransform
  DNNDeallocOpt.cpp
  eraseDummyConstants.cpp
  FunctionArgument.cpp
  mallocPoolOpt.cpp

  LINK_LIBS PUBLIC
  CDDNNOps
  CDDNNTransformOpFusing
  MLIRTransformUtils
)

add_subdirectory(opFusing)
