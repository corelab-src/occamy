# DNN Compiler

## Building an Image
This repository contains Dockerfile (occamy-docker) to build an onnx-mlir image with CUDA and CUDNN enabled. Follow these steps to build it.

1. Download [CUDNN 8.9.0 library for Linux](https://developer.nvidia.com/cudnn) that is compatible with CUDA 11.X. The file name should be `cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz`.

2. Move the file into `docker/` directory. `$ mv cudnn-linux-x86_64-8.9.0.131_cuda11-archive.tar.xz docker/`

3. Build an image with Dockerfile. `$ docker build . -f docker/Dockerfile.cuda.core-dnn --tag occamy-docker`

4. Run a container with a gpu option. `$ docker run --gpus 0 occamy-docker`

## Prerequisites

```
onnx == 1.13.1
onnxruntime == 1.14.1
scipy == 1.9.1
pandas == 2.0.1
tensorboardX == 2.6
numpy == 1.24.3
```

## Installation on UNIX

#### MLIR
Firstly, install MLIR (as a part of LLVM-Project):

``` bash
git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 21f4b84c456b471cc52016cf360e14d45f7f2960 && cd ..
```

``` bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

#### Occamy (core-dnn)

```bash
git clone --recursive https://github.com/corelab-src/occamy.git

# Export environment variables pointing to LLVM-Projects.
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

mkdir -p occamy/build && cd occamy/build
cmake -G Ninja ..
cmake --build . --target core-dnn -j32
```

##### LLVM and ONNX-MLIR CMake variables

The following CMake variables from LLVM and ONNX MLIR can be used when compiling ONNX MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.



