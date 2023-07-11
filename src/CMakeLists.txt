add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Tools)
add_subdirectory(Compiler)

get_property(CDWraps GLOBAL PROPERTY CORE_DNN_WRAPPERS)

# Main compiler file
add_core_dnn_executable(core-dnn
  core-dnn.cpp

  DEPENDS
  ${CDWraps}
  PyRuntime
  PyCompile
  PyCompileAndRuntime
  OMONNXOps

  LINK_LIBS PRIVATE
  onnx
  OMCompilerOptions
  OMCompilerUtils
  CoreDNNOptions
  CoreDNNUtils
  )

mlir_check_all_link_libraries(core-dnn)



