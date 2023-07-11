get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)
get_property(CDLibs GLOBAL PROPERTY CORE_DNN_LIBS)

add_core_dnn_library(CoreDNNOptions
  CoreDNNOptions.cpp

  EXCLUDE_FROM_CD_LIBS

  DEPENDS
  OMAccelerator
  ExternalUtil
  AcceleratorsInc

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}
  ${ONNX_MLIR_BIN_ROOT}/src/Compiler/Debug

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  MLIRIR
  OMAccelerator
)

add_core_dnn_library(CoreDNNPasses
  CoreDNNPasses.cpp

  EXCLUDE_FROM_CD_LIBS

  DEPENDS
  MLIRIR

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}
  ${ONNX_MLIR_BIN_ROOT}/src/Compiler/Debug

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  ${OMLibs}
  ${CDLibs}
  OMCompilerOptions
  CoreDNNOptions
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
)

add_core_dnn_library(CoreDNNUtils
  CoreDNNUtils.cpp

  EXCLUDE_FROM_CD_LIBS

  DEPENDS
  OMCompilerUtils
  ExternalUtil
  MLIRIR
  llc
  opt

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}
  ${ONNX_MLIR_BIN_ROOT}/src/Compiler/Debug

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  ${OMLibs}
  ${CDLibs}
  OMCompilerPasses
  CoreDNNPasses
  OMAccelerator
  OMInitAccelerators
  OMVersion

  LINK_LIBS PRIVATE
  OMCompilerUtils

  # Link LLVM libraries necessary to query which target architectures are
  # configured.
  LINK_COMPONENTS PRIVATE
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  MC
)

# CoreDNNUtils does not require cruntime or jniruntime to build, however,
# they are
# required for execution when using the EmitLib or EmitJNI options
add_dependencies(CoreDNNUtils cruntime)
if (ONNX_MLIR_ENABLE_JNI)
  add_dependencies(CoreDNNUtils jniruntime)
endif()

