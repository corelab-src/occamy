#ifndef __CONVERT_ONNX_TO_DNN_COMMON_CUDA_H__
#define __CONVERT_ONNX_TO_DNN_COMMON_CUDA_H__

int64_t calculateWorkspace(
    int64_t dimX[4], int64_t dimw[4],
    int64_t pads[4], int64_t strides[4], int64_t convAlgorithm, int64_t group);

int64_t calculateConvAlgo(
    int64_t dimX[4], int64_t dimw[4],
    int64_t pads[4], int64_t strides[4], int64_t group);

#endif
