
add_core_dnn_library(CDOpTransform
  ONNXConstantHoistingPass.cpp
  ONNXConstantAtUsePass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  OMSupport
  OMONNXRewrite
  OMShapeInference
  MLIRTransforms
)

