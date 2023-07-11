# SPDX-License-Identifier: Apache-2.0

# Match the minimum required version of LLVM and MLIR
cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Main project name
project(CORE-DNN)
set(CMAKE_BUILD_TYPE Debug)

# Enable warnings as errors
# Leverage the imported LLVM_ENABLE_WERROR for compiler logic
set(LLVM_ENABLE_WERROR ${CORE_DNN_ENABLE_WERROR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/bin)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/include)

set(CORE_DNN_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CORE_DNN_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR})

set(ONNX_MLIR_SRC_ROOT ${CORE_DNN_SRC_ROOT}/third_party/onnx-mlir)
set(ONNX_MLIR_BIN_ROOT ${CORE_DNN_BIN_ROOT}/third_party/onnx-mlir)

set(CORE_DNN_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CORE_DNN_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CORE_DNN_INCLUDE_PATH ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set(CORE_DNN_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific text for showing with version information.")

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(LLVM_ENABLE_PROJECTS mlir)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX;AMDGPU")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${ONNX_MLIR_BIN_ROOT}/third_party/onnx/)
include_directories(${ONNX_MLIR_SRC_ROOT}/third_party/onnx/)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_compile_definitions(ONNX_ML=1)

add_subdirectory(third_party/onnx-mlir)

include(MLIR.cmake)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

