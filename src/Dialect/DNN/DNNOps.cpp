//===------------------- DNNOps.cpp - DNN Operations -----------------===//

#include <iostream>
#include <queue>

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "src/Dialect/DNN/DNNOps.hpp"

using namespace mlir;

void DNNDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "src/Dialect/DNN/DNNOps.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "src/Dialect/DNN/DNNOps.cpp.inc"

#include "src/Dialect/DNN/DNNDialect.cpp.inc"
