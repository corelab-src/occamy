
//===------------ DNN Ops Header --------------===//
//===---------  XXX corelab Jaeho XXX -----------===//
//===--------------------------------------------===//

#ifndef __DNN_OPS_H__
#define __DNN_OPS_H__

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/DNN/DNNDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Dialect/DNN/DNNOps.hpp.inc"

#endif
