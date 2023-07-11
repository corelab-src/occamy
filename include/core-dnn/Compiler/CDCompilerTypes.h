/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CORE_DNN_CDCOMPILERTYPES_H
#define CORE_DNN_CDCOMPILERTYPES_H

#ifdef __cplusplus
namespace core_dnn {
#endif

// Emission target for core-dnn
enum EmissionTargetType {
  EmitONNXBasic,
  EmitONNXIR,
  EmitKrnl,
  EmitMLIR,
  EmitLLVMIR,
  EmitObj,
  EmitLib,
  EmitJNI,
};

// Hardware target for core-dnn
enum HardwareTargetType {
  x86,
  nvptx,
  nvptx_dnn,
};

#ifdef __cplusplus
} // namespace core_dnn
#endif

#endif /* CORE_DNN_CDCOMPILERTYPES_H */
