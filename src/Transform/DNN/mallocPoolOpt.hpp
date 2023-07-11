#ifndef DNN_MALLOC_POOL_H
#define DNN_MALLOC_POOL_H


#include "mlir/Analysis/Liveness.h"

typedef struct {
  int64_t start_addr;
  int64_t end_addr;
} freeBlock;

typedef struct {
  Operation* mallocPtr;
  int64_t start_addr;
  int64_t end_addr;
} allocBlock;

SmallVector<freeBlock> freePool = {{0, INT64_MAX-1}};
SmallVector<allocBlock> allocPool = {};

int64_t getMemRefSize(Operation* op) {
  Value memrefVal;
  if (isa<mlir::DNNMallocOp>(op))
    memrefVal = op->getResult(0);
  else if (isa<mlir::DNNDeallocOp>(op))
    memrefVal = op->getOperand(0);

  auto memRefType = convertToMemRefType(memrefVal.getType());
  auto shape = memRefType.getShape();
  int64_t opSize = 1;
  for (long unsigned int i=0; i<shape.size(); i++)
    opSize *= shape[i];
  opSize *= (int64_t)(memRefType.getElementType().getIntOrFloatBitWidth() / 8);
  //seungbin: set opSize as multiple of 32
  opSize = (opSize <= 0) ? 0 : ((opSize-1)/32+1)*32;
  return opSize; // Size in byte
}

void clearBlockVectors(void) {
  freePool.clear();
  freeBlock initBlock = {0, INT64_MAX-1};
  freePool.push_back({initBlock});
  allocPool.clear();
}

int findFirstFit (int64_t mallocSize) {
  for (long unsigned int fi=0; fi<freePool.size(); fi++) {
    if (freePool[fi].end_addr - freePool[fi].start_addr >= mallocSize)  {
      return fi;
    }
  }
  assert(0 && "Memory simulator out of memory");
}

int findBestFit (int64_t mallocSize) {
  int64_t tempFitSize = INT64_MAX;
  int tempFit = -1;
  for (long unsigned int fi=0; fi<freePool.size(); fi++) {
    int64_t freeSize = freePool[fi].end_addr - freePool[fi].start_addr;
    if (freeSize >= mallocSize && freeSize < tempFitSize)  {
      tempFitSize = freeSize;
      tempFit = fi;
    }
  }
  assert((tempFitSize != INT64_MAX) && "Memory simulator out of memory");
  return tempFit;
}

int64_t mallocSimul (Operation* mallocOp, int64_t mallocSize) {
  int fi = findFirstFit (mallocSize);
  // int fi = findBestFit (mallocSize);
  freeBlock freeblk = freePool[fi];

  allocBlock allocblk = {mallocOp, freeblk.start_addr, freeblk.start_addr+mallocSize-1};
  if (allocPool.size() == 0) {
    allocPool.insert(allocPool.begin(), allocblk);
    freePool[fi].start_addr += mallocSize;
    return allocblk.start_addr;
  } else {
    for (long unsigned int ai=0; ai<allocPool.size(); ai++) {
      if (allocPool[ai].start_addr < freeblk.start_addr) {
        if (ai == allocPool.size()-1) {
          allocPool.push_back(allocblk);
          freePool[fi].start_addr += mallocSize;
          return allocblk.start_addr;
        }
        continue;
      } else {
        allocPool.insert(allocPool.begin()+ai, allocblk);
        freePool[fi].start_addr += mallocSize;
        return allocblk.start_addr;
      }
    }
  }
  return -1;
}

bool deallocSimul (Operation* mallocOpPtr) {
  for (long unsigned int ai=0; ai<allocPool.size(); ai++) {
    allocBlock allocblk = allocPool[ai];
    allocBlock deallocblk;
    if (allocblk.mallocPtr == mallocOpPtr) {
      deallocblk = allocPool[ai];
      allocPool.erase(allocPool.begin()+ai);
      for (long unsigned int fi=0; fi<freePool.size(); fi++) {
        if (freePool[fi].end_addr == deallocblk.start_addr-1) {
          freePool[fi].end_addr = deallocblk.end_addr;
          if (fi < freePool.size()-1) {
            if (freePool[fi+1].start_addr == freePool[fi].end_addr+1) {
              freePool[fi+1].start_addr = freePool[fi].start_addr;
              freePool.erase(freePool.begin()+fi);
            }
          }
          return true;
        } else if (freePool[fi].start_addr == deallocblk.end_addr+1) {
          freePool[fi].start_addr = deallocblk.start_addr;
          if (fi > 0) {
            if (freePool[fi-1].end_addr == freePool[fi].start_addr-1) {
              freePool[fi-1].end_addr = freePool[fi].end_addr;
              freePool.erase(freePool.begin()+fi);
            }
          }
          return true;
        }
      }
      for (long unsigned int fi=0; fi<freePool.size(); fi++) {
        freeBlock newFree = {deallocblk.start_addr, deallocblk.end_addr};
        if (freePool[fi].start_addr > deallocblk.end_addr) {
          freePool.insert(freePool.begin()+fi, newFree);
          return true;
        } else if ((fi==freePool.size()-1) &&
            freePool[fi].start_addr < deallocblk.end_addr) {
          freePool.push_back(newFree);
          return true;
        }
      }
    }
  }
  return false;
}

Operation* backupInterTensor (Operation* divOp, Operation* mallocOp) {
  auto loc = divOp->getLoc();
  OpBuilder builder(divOp);
  auto int32Ty = builder.getIntegerType(32);
  builder.setInsertionPoint(divOp);

  // alloc hostmemory and copy to host
  auto memSize = getMemRefSize(mallocOp);
  auto sizeConst = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(memSize));
  MemRefType memrefType =
    convertToMemRefType(mallocOp->getResult(0).getType());
  auto alloc = builder.create<memref::AllocOp>(loc, memrefType);
  builder.create<DNNMemcpyOp>(loc, int32Ty, alloc, mallocOp->getResult(0),
      sizeConst, builder.getI32IntegerAttr(2)); // Type = device to host
  Operation* prevDealloc;
  for (Operation* user: mallocOp->getUsers()) {
    if (isa<DNNDeallocOp>(user)) {
      prevDealloc = user;
      break;
    }
  }
  builder.create<DNNDeallocOp>(loc, int32Ty, mallocOp->getResult(0));

  // copy back to device
  builder.setInsertionPointAfter(divOp);
  auto mallocBack = builder.create<DNNMallocOp>(loc, memrefType, sizeConst);
  builder.create<DNNMemcpyOp>(loc, int32Ty, mallocBack, alloc,
      sizeConst, builder.getI32IntegerAttr(1)); // Type = host to device
  prevDealloc->setOperand(0, mallocBack);

  // find use of old malloc and replace with new malloc
  Operation* funcOpPtr = divOp->getParentOp();
  func::FuncOp funcOp = dyn_cast<func::FuncOp>(*funcOpPtr);
  for (Operation* opIt = divOp; opIt != (&funcOp.back().back()); opIt = opIt->getNextNode()) {
    int operIdx = 0;
    for (Value operand : opIt->getOperands()) {
      if (operand.getDefiningOp() == mallocOp) {
        opIt->setOperand(operIdx, mallocBack);
        break;
      }
      operIdx++;
    }
  }
  SmallVector<Operation*, 1> returnOp;
  /* returnOp stores the operations whose return value
   * sould stored in mallocBack operation.  */
  for (Operation* opIt = mallocOp; opIt != (&funcOp.back().back()); opIt = opIt->getNextNode()) {
    unsigned numOper = opIt->getNumOperands();
    if (numOper != 0) {
      if (opIt->getOperand(numOper-1).getDefiningOp() == mallocOp) {
        returnOp.emplace_back(opIt);
      }
    }
  }
  for (int i = 0; i < (int)returnOp.size(); i++) {
    Operation* targetOp = returnOp[i];
    for (Operation* opIt = divOp; opIt != (&funcOp.back().back()); opIt = opIt->getNextNode()) {
      int operIdx = 0;
      for (Value operand : opIt->getOperands()) {
        if (operand.getDefiningOp() == targetOp) {
          opIt->setOperand(operIdx, mallocBack);
          break;
        }
        operIdx++;
      }
    }
  }
  return alloc;
}

void updateMaxAllocByte (OpBuilder builder, int64_t& maxAllocByte, arith::ConstantOp& poolSizeConst) {
  if (maxAllocByte < freePool[freePool.size()-1].start_addr) {
    maxAllocByte = freePool[freePool.size()-1].start_addr;
  }
  poolSizeConst.getOperation()->setAttr("value",
      builder.getI64IntegerAttr(maxAllocByte));
}

void replaceUseForMemOffsetOp (SmallVector<Operation*, 1> &vec) {
  for (long unsigned int i=0; i<vec.size(); i=i+2) {
    vec[i]->replaceAllUsesWith(vec[i+1]);
    auto constOp = vec[i]->getOperand(0).getDefiningOp();
    vec[i]->erase();
    if (constOp->use_empty())
      constOp->erase();
  }
  vec.clear();
}

void eraseAndFreeDeallocVec (SmallVector<Operation*, 1> &vec) {
  for (long unsigned int i=0; i<vec.size(); i++) {
    Operation* deallocOp = vec[i];
    vec[i]->erase();
  }
  vec.clear();
}

Operation* generateMemPoolInit (OpBuilder builder, int64_t& maxAllocByte, Operation* anchorOp) {
  auto loc = anchorOp->getLoc();

  auto memPoolSize = builder.create<arith::ConstantOp>(loc,
      builder.getI64IntegerAttr(maxAllocByte));
  SmallVector<int64_t, 1> poolShape;
  poolShape.push_back(1);
  auto poolMemRefType = MemRefType::get(poolShape, builder.getF32Type());
  auto memPoolInitOp = builder.create<mlir::DNNMemPoolInitOp>(loc,
      poolMemRefType, memPoolSize);
  memPoolSize.getOperation()->moveAfter(anchorOp);
  memPoolInitOp.getOperation()->moveAfter(memPoolSize);

  return memPoolInitOp;
}

void insertPoolDealloc (OpBuilder builder, Operation* poolInitOp) {
  auto loc = poolInitOp->getLoc();
  Operation* funcOpPtr = poolInitOp->getParentOp();
  func::FuncOp funcOp = dyn_cast<func::FuncOp>(*funcOpPtr);

  Value memPoolVal = poolInitOp->getResult(0);
  auto int32Ty = builder.getIntegerType(32);
  DNNDeallocOp dealloc = builder.create<DNNDeallocOp>(loc, int32Ty, memPoolVal);
  dealloc.getOperation()->moveBefore(&funcOp.back().back());
  return;
}

#endif // DNN_MALLOC_POOL_H
