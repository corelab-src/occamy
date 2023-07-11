/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.cpp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

#include "ExternalUtil.hpp"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Compiler/CoreDNNOptions.hpp"
#include "src/Compiler/CoreDNNPasses.hpp"
#include "src/Compiler/CoreDNNUtils.hpp"
#include "src/Version/Version.hpp"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/DNN/DNNOps.hpp"

#define DEBUG_TYPE "corednn_utils"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

namespace core_dnn {

onnx_mlir::EmissionTargetType emissionCOREToONNX(core_dnn::EmissionTargetType coreEmission) {
  switch((int)coreEmission) {
      case 10:
        return onnx_mlir::EmitJNI;
      case 9:
        return onnx_mlir::EmitLib;
      case 8:
        return onnx_mlir::EmitObj;
      case 7:
        return onnx_mlir::EmitLLVMIR;
      case 6:
      case 5:
      case 4:
      case 3:
      case 2:
        return onnx_mlir::EmitMLIR;
      case 1:
        return onnx_mlir::EmitONNXIR;
      case 0:
        return onnx_mlir::EmitONNXBasic;
    }
  assert(0 && "CoreDNNUtils emissionCOREToONNX : Unreachable!");
}

core_dnn::EmissionTargetType emissionONNXToCORE(onnx_mlir::EmissionTargetType onnxEmission) {
  switch((int)onnxEmission) {
      case 6:
        return core_dnn::EmitJNI;
      case 5:
        return core_dnn::EmitLib;
      case 4:
        return core_dnn::EmitObj;
      case 3:
        return core_dnn::EmitLLVMIR;
      case 2:
        return core_dnn::EmitMLIR;
      case 1:
        return core_dnn::EmitONNXIR;
      case 0:
        return core_dnn::EmitONNXBasic;
    }
  assert(0 && "CoreDNNUtils emissionONNXToCORE : Unreachable!");
}

// Return the vendor name if specified during make processing or the default.
std::string getVendorName() {
  return "CORE-DNN";
}

// Make a function that forces preserving all files using the runtime arguments
// and/or the overridePreserveFiles enum.
enum class KeepFilesOfType { All, MLIR, LLVMIR, Bitcode, Object, None };

// Value below override at compile time by effectively setting the requested
// flags.
static constexpr KeepFilesOfType overridePreserveFiles = KeepFilesOfType::None;

static bool keepFiles(KeepFilesOfType preserve) {
  // When wanting to preserve all files, do it regardles of isBitcode.
  if (overridePreserveFiles == KeepFilesOfType::All)
    return true;
  // When file is bitcode, check the runtime flag preserveBitcode.
  switch (preserve) {
  case KeepFilesOfType::Bitcode:
    return overridePreserveFiles == KeepFilesOfType::Bitcode || preserveBitcode;
  case KeepFilesOfType::LLVMIR:
    return overridePreserveFiles == KeepFilesOfType::LLVMIR || preserveLLVMIR;
  case KeepFilesOfType::MLIR:
    return overridePreserveFiles == KeepFilesOfType::MLIR || preserveMLIR;
  case KeepFilesOfType::Object:
    // Currently no option, enable using the overridePreserveFiles enum.
    return overridePreserveFiles == KeepFilesOfType::Object;
  default:
    // All, None should not be used in the parameter
    llvm_unreachable("illegal KeepFilesOfType enum value");
  }
  return false;
}

static std::string getExecPath() {
  // argv0 is only used as a fallback for rare environments
  // where /proc isn't mounted and mainExecAddr is only needed for
  // unknown unix-like platforms
  auto execPath = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (execPath.empty()) {
    llvm::errs()
        << "Warning: Could not find path to current executable, falling "
           "back to default install path: "
        << kExecPath << "\n";
    return kExecPath;
  }
  return execPath;
}

// Directory contains all the libraries, jars, etc. that are necessary for
// running onnx-mlir. It's resolved in the following order:
//
//   - if ONNX_MLIR_LIBRARY_PATH is set, use it, otherwise
//   - get path from where onnx-mlir is run, if it's of the form
//     /foo/bar/bin/onnx-mlir,
//     the runtime directory is /foo/bar/lib (note that when onnx-mlir is
//     installed system wide, which is typically /usr/local/bin, this will
//     correctly resolve to /usr/local/lib), but some systems still have
//     lib64 so we check that first. If neither exists, then
//   - use CMAKE_INSTALL_PREFIX/lib, which is typically /usr/local/lib
//
// We now explicitly set CMAKE_INSTALL_LIBDIR to lib so we don't have
// to deal with lib64 anymore.
static std::string getLibraryPath() {
  const auto &envDir = getEnvVar("ONNX_MLIR_LIBRARY_PATH");
  if (envDir && llvm::sys::fs::exists(envDir.value()))
    return envDir.value();

  std::string execDir = llvm::sys::path::parent_path(getExecPath()).str();
  if (llvm::sys::path::stem(execDir).str().compare("bin") == 0) {
    std::string p = execDir.substr(0, execDir.size() - 3);
    if (llvm::sys::fs::exists(p + "lib"))
      return p + "lib";
  }

  llvm::SmallString<8> instDir(kInstPath);
  llvm::sys::path::append(instDir, "lib");
  return llvm::StringRef(instDir).str();
}
} // namespace core_dnn

namespace core_dnn {
// =============================================================================
// Methods for compiling and file processing.

static void loadMLIR(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<ModuleOp> &module) {
  // Handle '.mlir' input to the ONNX-MLIR frontend.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    exit(1);
  }
}

// Tailor LLVMIR to add features that cannot be done with MLIR LLVMIR.
static void tailorLLVMIR(llvm::Module &llvmModule) {
  llvm::LLVMContext &ctx = llvmModule.getContext();
  // Emit metadata "zos_le_char_mode" for z/OS. Use EBCDIC codepage by default.
  if (llvm::Triple(getTargetTripleOption()).isOSzOS()) {
    StringRef charModeKey = "zos_le_char_mode";
    if (!llvmModule.getModuleFlag(charModeKey)) {
      auto val = llvm::MDString::get(ctx, "ebcdic");
      llvmModule.addModuleFlag(llvm::Module::Error, charModeKey, val);
    }
  }

  // Emit the onnx-mlir version as llvm.ident metadata.
  llvm::NamedMDNode *identMetadata =
      llvmModule.getOrInsertNamedMetadata("llvm.ident");
  llvm::Metadata *identNode[] = {
      llvm::MDString::get(ctx, getOnnxMlirFullVersion())};
  identMetadata->addOperand(llvm::MDNode::get(ctx, identNode));

#ifdef PRODUCT_VERSION_MAJOR
  int32_t ProductVersion = PRODUCT_VERSION_MAJOR;
  llvmModule.addModuleFlag(
      llvm::Module::Warning, "Product Major Version", ProductVersion);
#endif
#ifdef PRODUCT_VERSION_MINOR
  int32_t ProductRelease = PRODUCT_VERSION_MINOR;
  llvmModule.addModuleFlag(
      llvm::Module::Warning, "Product Minor Version", ProductRelease);
#endif
#ifdef PRODUCT_VERSION_PATCH
  int32_t ProductPatch = PRODUCT_VERSION_PATCH;
  llvmModule.addModuleFlag(
      llvm::Module::Warning, "Product Patchlevel", ProductPatch);
#endif
#ifdef PRODUCT_ID
  llvmModule.addModuleFlag(llvm::Module::Warning, "Product Id",
      llvm::MDString::get(ctx, PRODUCT_ID));
#endif

  // Annotate functions to be accessible from DLL on Windows.
#ifdef _WIN32
  SmallVector<StringRef, 4> exportedFuncs;
  // Signature functions.
  exportedFuncs.emplace_back(StringRef("omInputSignature"));
  exportedFuncs.emplace_back(StringRef("omOutputSignature"));
  exportedFuncs.emplace_back(StringRef("omQueryEntryPoints"));
  // Entry point funtions.
  if (llvm::GlobalVariable *GV =
          llvmModule.getNamedGlobal(StringRef("_entry_point_arrays"))) {
    if (GV->isConstant() && GV->hasDefinitiveInitializer()) {
      llvm::Constant *initializer = GV->getInitializer();
      llvm::ArrayType *AT = dyn_cast<llvm::ArrayType>(initializer->getType());
      for (uint64_t i = 0; i < AT->getNumElements() - 1; ++i) {
        llvm::GlobalVariable *entryGV = llvmModule.getNamedGlobal(
            StringRef("_entry_point_" + std::to_string(i)));
        if (entryGV->isConstant()) {
          llvm::ConstantDataSequential *entry =
              dyn_cast<llvm::ConstantDataSequential>(entryGV->getInitializer());
          exportedFuncs.emplace_back(entry->getAsCString());
        }
      }
    }
  }
  for (StringRef funcName : exportedFuncs)
    if (llvm::GlobalValue *GV = llvmModule.getNamedValue(funcName)) {
      GV->setDSOLocal(true);
      GV->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    }
#endif
}

// Write LLVM optimized bitcode.
// Returns 0 on success, error code on failure.
static int genLLVMBitcode(const mlir::OwningOpRef<ModuleOp> &module,
    std::string outputNameNoExt, std::string optimizedBitcodeNameWithExt) {
  std::error_code error;

  // Write bitcode to a file.
  std::string unoptimizedBitcodeNameWithExt =
      outputNameNoExt + ".unoptimized.bc";
  llvm::FileRemover unoptimizedBitcodeRemover(
      unoptimizedBitcodeNameWithExt, !keepFiles(KeepFilesOfType::Bitcode));

  // outputNameNoExt might contain a directory, which must exist.
  // Otherwise, a "No such file or directory" error will be returned.
  llvm::raw_fd_ostream moduleBitcodeStream(
      unoptimizedBitcodeNameWithExt, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << unoptimizedBitcodeNameWithExt << ": " << error.message()
                 << "\n";
    return InvalidTemporaryFileAccess;
  }

  llvm::LLVMContext llvmContext;
  mlir::registerLLVMDialectTranslation(*(module.get().getContext()));
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVMIR.\n";
    return CompilerFailureInMLIRToLLVM;
  }

  // Tailor LLVMIR to add features that cannot be done with MLIR LLVMIR.
  tailorLLVMIR(*llvmModule);

  // Write LLVMIR to a file.
  std::string llvmirNameWithExt = outputNameNoExt + ".ll";
  llvm::FileRemover llvmirRemover(
      llvmirNameWithExt, !keepFiles(KeepFilesOfType::LLVMIR));
  llvm::raw_fd_ostream moduleLLVMIRStream(
      llvmirNameWithExt, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << llvmirNameWithExt << ": " << error.message() << "\n";
    return InvalidTemporaryFileAccess;
  }
  llvmModule->print(moduleLLVMIRStream, nullptr);
  moduleLLVMIRStream.flush();

  // Write unoptimized bitcode to a file.
  llvm::WriteBitcodeToFile(*llvmModule, moduleBitcodeStream);
  moduleBitcodeStream.flush();

  // Use the LLVM's 'opt' command to optimize the bitcode.
  std::string optPath = getToolPath("opt", kOptPath);
  Command optBitcode(/*exePath=*/optPath);
  int rc = optBitcode.appendStr(getOptimizationLevelOption())
               .appendStr(getTargetTripleOption())
               .appendStr(getTargetArchOption())
               .appendStr(getTargetCPUOption())
               .appendList(getXoptOption())
               .appendStr(getLLVMOption())
               .appendList({"-o", optimizedBitcodeNameWithExt})
               .appendStr(unoptimizedBitcodeNameWithExt)
               .exec();
  return rc != 0 ? CompilerFailureInLLVMOpt : CompilerSuccess;
}

// Compile LLVM bitcode to object file.
// Return 0 on success, error code on failure.
static int genModelObject(
    std::string bitcodeNameWithExt, std::string &modelObjNameWithExt) {

  std::string llcPath = getToolPath("llc", kLlcPath);
  Command llvmToObj(/*exePath=*/llcPath);
  int rc = llvmToObj.appendStr(getOptimizationLevelOption())
               .appendStr(getTargetTripleOption())
               .appendStr(getTargetArchOption())
               .appendStr(getTargetCPUOption())
               .appendList(getXllcOption())
               .appendStr(getLLVMOption())
               .appendStr("-filetype=obj")
               .appendStr("-relocation-model=pic")
               .appendList({"-o", modelObjNameWithExt})
               .appendStr(bitcodeNameWithExt)
               .exec();
  return rc != 0 ? CompilerFailureInLLVMToObj : CompilerSuccess;
}

// Return 0 on success, error code on failure.
static int genJniObject(const mlir::OwningOpRef<ModuleOp> &module,
    std::string jniSharedLibPath, std::string jniObjPath) {
  Command ar(/*exePath=*/kArPath);
  int rc = ar.appendStr("x")
               // old version of ar does not support --output so comment out
               // for now and use the optional wdir for exec() to get around
               // the problem.
               //.appendStr("--output")
               //.appendStr(llvm::sys::path::parent_path(jniObjPath).str())
               .appendStr(jniSharedLibPath)
               .appendStr(llvm::sys::path::filename(jniObjPath).str())
               .exec(llvm::sys::path::parent_path(jniObjPath).str());
  return rc != 0 ? CompilerFailureInGenJniObj : CompilerSuccess;
}

// Link everything into a shared object.
// Return 0 on success, error code on failure.
static int genSharedLib(std::string sharedLibNameWithExt,
    std::vector<std::string> opts, std::vector<std::string> objs,
    std::vector<std::string> libs, std::vector<std::string> libDirs) {

#ifdef _WIN32
  std::vector<std::string> outputOpt = {"/Fe:" + sharedLibNameWithExt};
  // link has to be before libpath since they need to be passed through to the
  // linker
  std::vector<std::string> sharedLibOpts = {"/LD", "/link", "/NOLOGO"};

  llvm::for_each(libs, [](std::string &lib) { lib = lib + ".lib"; });
  llvm::for_each(libDirs,
      [](std::string &libDir) { libDir = "/libpath:\"" + libDir + "\""; });
#else
  std::vector<std::string> outputOpt = {"-o", sharedLibNameWithExt};
  std::vector<std::string> sharedLibOpts = {"-shared", "-fPIC"};
  llvm::for_each(libs, [](std::string &lib) { lib = "-l" + lib; });
  llvm::for_each(libDirs, [](std::string &libDir) { libDir = "-L" + libDir; });
#endif

  Command link(kCxxPath);
  int rc = link.appendList(opts)
               .appendList(objs)
               .appendList(outputOpt)
               .appendList(sharedLibOpts)
               .appendList(libDirs)
               .appendList(libs)
               .exec();
  return rc != 0 ? CompilerFailureInObjToLib : CompilerSuccess;
}

// Create jar containing java runtime and model shared library (which includes
// jni runtime).
// Return 0 on success, error code on failure.
static int genJniJar(const mlir::OwningOpRef<ModuleOp> &module,
    std::string modelSharedLibPath, std::string modelJniJarPath) {
  llvm::SmallString<8> libraryPath(getLibraryPath());
  llvm::sys::path::append(libraryPath, "javaruntime.jar");
  std::string javaRuntimeJarPath = llvm::StringRef(libraryPath).str();

  // Copy javaruntime.jar to model jar.
  llvm::sys::fs::copy_file(javaRuntimeJarPath, modelJniJarPath);

  // Add shared library to model jar.
  Command jar(kJarPath);
  int rc =
      jar.appendStr("uf")
          .appendStr(modelJniJarPath)
          .appendStr("-C")
          .appendStr(llvm::sys::path::parent_path(modelSharedLibPath).str())
          .appendStr(llvm::sys::path::filename(modelSharedLibPath).str())
          .exec();
  return rc != 0 ? CompilerFailureInGenJni : CompilerSuccess;
}

// Return 0 on success, error code on failure
static int compileModuleToObject(const mlir::OwningOpRef<ModuleOp> &module,
    std::string outputNameWithoutExt, std::string &objectNameWithExt) {
  std::string bitcodeNameWithExt = outputNameWithoutExt + ".bc";
  int rc = genLLVMBitcode(module, outputNameWithoutExt, bitcodeNameWithExt);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover bitcodeRemover(
      bitcodeNameWithExt, !keepFiles(KeepFilesOfType::Bitcode));
  objectNameWithExt = getTargetFilename(outputNameWithoutExt, onnx_mlir::EmitObj);
  return genModelObject(bitcodeNameWithExt, objectNameWithExt);
}

// Return 0 on success, error code on failure
static int compileModuleToSharedLibrary(
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputNameNoExt,
    std::string &libNameWithExt, HardwareTargetType hardwareTarget) {
  std::string modelObjNameWithExt;
  int rc = compileModuleToObject(module, outputNameNoExt, modelObjNameWithExt);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover modelObjRemover(
      modelObjNameWithExt, !keepFiles(KeepFilesOfType::Object));

  libNameWithExt = getTargetFilename(outputNameNoExt, onnx_mlir::EmitLib);

  // Enable other hardware libraries.
  std::vector<std::string> libDirs = {getLibraryPath()};
  if (hardwareTarget == nvptx_dnn) {
    libDirs.push_back("/usr/local/cuda/lib64");
  }

  return genSharedLib(libNameWithExt, {}, {modelObjNameWithExt},
      getCompilerConfig(CCM_SHARED_LIB_DEPS), libDirs);
}

// Return 0 on success, error code on failure
static int compileModuleToJniJar(
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputNameNoExt) {
  std::string modelObjNameWithExt;
  int rc = compileModuleToObject(module, outputNameNoExt, modelObjNameWithExt);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover modelObjRemover(
      modelObjNameWithExt, !keepFiles(KeepFilesOfType::Object));

  StringRef outputDir = llvm::sys::path::parent_path(outputNameNoExt);
  if (outputDir.empty())
    outputDir = StringRef(".");

  std::string jniSharedLibPath = getLibraryPath() + "/libjniruntime.a";

  llvm::SmallString<8> jniObjDir(outputDir);
  llvm::sys::path::append(jniObjDir, "jnidummy.c.o");
  std::string jniObjPath = llvm::StringRef(jniObjDir).str();

  rc = genJniObject(module, jniSharedLibPath, jniObjPath);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover jniObjRemover(
      jniObjPath, !keepFiles(KeepFilesOfType::Object));

  llvm::SmallString<8> jniLibDir(outputDir);
  llvm::sys::path::append(jniLibDir, "libmodel");
  std::string jniLibBase = llvm::StringRef(jniLibDir).str();

#if defined(__APPLE__) && defined(__clang__)
#define NOEXECSTACK                                                            \
  {}
#else
#define NOEXECSTACK                                                            \
  { "-z", "noexecstack" }
#endif
  std::string modelSharedLibPath = getTargetFilename(jniLibBase, onnx_mlir::EmitLib);
  rc = genSharedLib(modelSharedLibPath, NOEXECSTACK,
      {modelObjNameWithExt, jniObjPath}, getCompilerConfig(CCM_SHARED_LIB_DEPS),
      {getLibraryPath()});
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover modelSharedLibRemover(
      modelSharedLibPath, !keepFiles(KeepFilesOfType::Object));

  std::string modelJniJarPath = getTargetFilename(outputNameNoExt, onnx_mlir::EmitJNI);
  return genJniJar(module, modelSharedLibPath, modelJniJarPath);
}

void registerDialects(mlir::MLIRContext &context) {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::AffineDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::shape::ShapeDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::ONNXDialect>();
  context.getOrLoadDialect<mlir::KrnlDialect>();
  context.getOrLoadDialect<mlir::DNNDialect>();
}

namespace {
std::string dirName(StringRef inputFilename) {
  llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
  llvm::sys::path::remove_filename(path);
  return std::string(path.data(), path.size());
}
} // namespace

// Return 0 on success, error code on failure.
static int emitOutputFiles(std::string outputNameNoExt,
    core_dnn::EmissionTargetType emissionTarget, HardwareTargetType hardwareTarget,
    mlir::MLIRContext &context, mlir::OwningOpRef<ModuleOp> &module) {

  VerboseOutput = true;

  // For EmitONNXIR and EmitMLIR the constant value are embedded in the code
  // thus making the code hard to read. These values can be elided by emitting
  // two versions of the same source code:
  // (1) a version with all the constant values included meant for being passed
  //     back to onnx-mlir for further processing and stored in:
  //
  //     <name>.onnx.mlir
  //
  // (2) a version without constants meant for being inspected by users and
  //     stored in:
  //
  //     <name>.tmp
  //
  // In the case of the LLVM Dialect IR the constant values are grouped
  // outside the function code at the beginning of the file in which case the
  // elision of these constants is not strictly required. Elision is also not
  // necessary when emitting the .bc file.
  switch (emissionTarget) {
  case EmitObj: {
    std::string modelObjNameWithExt;
    int rc =
        compileModuleToObject(module, outputNameNoExt, modelObjNameWithExt);
    if (rc != CompilerSuccess)
      return rc;
    if (keepFiles(KeepFilesOfType::MLIR)) {
      rc = outputCode(module, outputNameNoExt + ".llvm.mlir");
      if (rc != CompilerSuccess)
        return rc;
    }
    if (VerboseOutput)
      printf(
          "Object file %s has been compiled.\n", modelObjNameWithExt.c_str());
  } break;
  case EmitLib: {
    if (hardwareTarget == nvptx_dnn) {
      addCompilerConfig(CCM_SHARED_LIB_DEPS, {"cd_dnn_wrapper", "cudart", "cudnn", "cublas",  "cublasLt"});
    }
    addCompilerConfig(CCM_SHARED_LIB_DEPS, {"cruntime"});
    std::string sharedLibNameWithExt;
    int rc = compileModuleToSharedLibrary(
        module, outputNameNoExt, sharedLibNameWithExt, hardwareTarget);
    if (rc != CompilerSuccess)
      return rc;
    if (keepFiles(KeepFilesOfType::MLIR)) {
      rc = outputCode(module, outputNameNoExt + ".llvm.mlir");
      if (rc != CompilerSuccess)
        return rc;
    }
    if (VerboseOutput)
      printf("Shared library %s has been compiled.\n",
          sharedLibNameWithExt.c_str());
  } break;
  case EmitJNI: {
    addCompilerConfig(CCM_SHARED_LIB_DEPS, {"jniruntime", "cruntime"});
    int rc = compileModuleToJniJar(module, outputNameNoExt);
    if (rc != CompilerSuccess)
      return rc;
    if (keepFiles(KeepFilesOfType::MLIR)) {
      rc = outputCode(module, outputNameNoExt + ".llvm.mlir");
      if (rc != CompilerSuccess)
        return rc;
    }
    if (VerboseOutput)
      printf(
          "JNI archive %s.jar has been compiled.\n", outputNameNoExt.c_str());
  } break;
  default: {
    // Emit the version with all constants included.
    onnx_mlir::EmissionTargetType onnxEmitTarget = emissionCOREToONNX(emissionTarget);
    std::string ouputNameWithExt =
        getTargetFilename(outputNameNoExt, onnxEmitTarget);
    int rc = outputCode(module, ouputNameWithExt);
    if (VerboseOutput)
      printf("Full MLIR code written to: \n\t%s\n\n", ouputNameWithExt.c_str());
    if (rc != CompilerSuccess)
      return rc;

    // Elide element attributes if larger than 100.
    if (emissionTarget == EmitONNXBasic || emissionTarget == EmitONNXIR ||
        emissionTarget == EmitMLIR) {
      std::string tempNameWithExt = outputNameNoExt + ".tmp";
      int rc = outputCode(module, tempNameWithExt, /*largeElementLimit=*/100);
      if (VerboseOutput) {
        printf("Constant-free MLIR Code written to: \n\t%s\n\n",
            tempNameWithExt.c_str());
        printf("Use:\n\t%s\nto continue lowering the code to other dialects.\n",
            ouputNameWithExt.c_str());
      }
      if (rc != CompilerSuccess)
        return rc;
    }
  }
  }
  return CompilerSuccess;
} // end anonymous namespace

// Get the LLVM Target object corresponding to the target triple (if valid).
static const llvm::Target *getLLVMTarget(
    const std::string &targetTriple, const Location &loc) {
  std::string error;
  const llvm::Target *LLVMTarget =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!LLVMTarget) {
    emitError(loc, Twine("Target architecture is unknown: ") + error);
    return nullptr;
  }

  return LLVMTarget;
}

static std::string getTargetTriple() {
  return (mtriple != "") ? mtriple.getValue() : kDefaultTriple;
}
static std::string getTargetCpu() {
  return (mcpu != "") ? mcpu.getValue() : "";
}

/// Return the module datalayout string. The datalayout string is determined
/// by creating a target machine using the target triple and target cpu.
static std::string getDataLayout(const Location &loc) {
  const std::string targetTriple = getTargetTriple();
  const std::string targetCpu = getTargetCpu();
  const llvm::Target &LLVMTarget = *getLLVMTarget(targetTriple, loc);
  llvm::TargetOptions ops;
  auto targetMachine =
      std::unique_ptr<llvm::TargetMachine>{LLVMTarget.createTargetMachine(
          targetTriple, targetCpu, "" /*features*/, ops, std::nullopt)};
  if (!targetMachine) {
    emitError(loc, "failed to create target machine");
    return nullptr;
  }

  const llvm::DataLayout &dl = targetMachine->createDataLayout();
  std::string dataLayoutString = dl.getStringRepresentation();
  assert(dataLayoutString != "" && "Expecting a valid target datalayout");

  return dataLayoutString;
}

// Return 0 on success, error code on failure.
static int setupModule(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt) {
  // Initialize the targets support for all targets LLVM was configured for.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Set the module target triple and datalayout.
  Operation &moduleOp = *(module->getOperation());
  Location loc = moduleOp.getLoc();
  moduleOp.setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
      StringAttr::get(&context, getTargetTriple()));
  moduleOp.setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
      StringAttr::get(&context, getDataLayout(loc)));

  // Set the module target accelerators.
  SmallVector<Attribute, 2> accelsAttr;
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
    std::ostringstream versionNumber;
    versionNumber << std::hex << accel->getVersionNumber();
    std::string accelStr = accel->getName() + "-0x" + versionNumber.str();
    accelsAttr.emplace_back(StringAttr::get(&context, accelStr));
  }
  if (!accelsAttr.empty())
    moduleOp.setAttr("onnx-mlir.accels", ArrayAttr::get(&context, accelsAttr));

  if (keepFiles(KeepFilesOfType::MLIR)) {
    std::string mlirNameWithExt = outputNameNoExt + ".input.mlir";
    int rc = outputCode(module, mlirNameWithExt);
    if (rc != CompilerSuccess)
      return rc;
    module.release();
    loadMLIR(mlirNameWithExt, context, module);
  }
  return CompilerSuccess;
}

static int emitOutput(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    mlir::PassManager &pm, core_dnn::EmissionTargetType emissionTarget,
    HardwareTargetType hardwareTarget) {
  if (printIR) {
    mlir::OpPrintingFlags flags;
    if (preserveLocations)
      flags.enableDebugInfo();
    module->print(llvm::outs(), flags);
    return CompilerSuccess;
  }
  return emitOutputFiles(outputNameNoExt, emissionTarget, hardwareTarget, context, module);
}

// Return 0 on success, error code on error.
int compileModule(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    core_dnn::EmissionTargetType emissionTarget,
    HardwareTargetType hardwareTarget) {
  // Initialize accelerator(s) if required.
  if (!maccel.empty())
    onnx_mlir::accel::initAccelerators(maccel);

  int rc = setupModule(module, context, outputNameNoExt);
  if (rc != CompilerSuccess)
    return rc;

  mlir::PassManager pm(module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);

  InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= ONNXLevel && emissionTarget >= EmitONNXIR) {
    addONNXToMLIRPasses(pm, /*target CPU*/ maccel.empty());

    if (onnxConstHoisting) {
      pm.addNestedPass<func::FuncOp>(createONNXConstantHoistingPass());
    } else if (onnxConstAtUse) {
      pm.addNestedPass<func::FuncOp>(createONNXConstantAtUsePass());
    }
  }

  if (emissionTarget >= EmitMLIR) {
    if (hardwareTarget == nvptx_dnn) {
      std::cout << "ONNX dialect --> DNN dialect" << std::endl;
      addONNXToDNNPasses(pm);

      if (dnnKernelFusion)
        pm.addPass(createfuseConvBiasActivPass());
      if (dnnDeallocOpt)
        pm.addPass(createDNNDeallocOptPass());
      if (dnnmallocPoolOpt)
        pm.addPass(createmallocPoolOptPass());

      std::cout << "ONNX dialect --> Krnl dialect" << std::endl;
      addONNXToKrnlPasses(pm, OptimizationLevel, /*enableCSE*/ true,
          instrumentONNXSignature, ONNXOpStats);

    } else {
      if (inputIRLevel <= ONNXLevel) {
        std::cout << "ONNX dialect --> Krnl dialect" << std::endl;
        addONNXToKrnlPasses(pm, OptimizationLevel, /*enableCSE*/ true,
            instrumentONNXSignature, ONNXOpStats);
      }
      if (inputIRLevel <= MLIRLevel) {
        std::cout << "Krnl dialect --> Affine dialect" << std::endl;
        addKrnlToAffinePasses(pm);
      }
    }
  }

  if (inputIRLevel <= LLVMLevel && emissionTarget >= EmitLLVMIR) {
    if (hardwareTarget == nvptx_dnn) {
      std::cout << "DNN dialect --> LLVM dialect" << std::endl;
      addDNNToLLVMPasses(pm);
      std::cout << "Krnl dialect --> LLVM dialect" << std::endl;
      addKrnlToLLVMPasses(pm, /*enableCSE=*/true, verifyInputTensors);
    } else {
      std::cout << "Krnl dialect --> LLVM dialect" << std::endl;
      addKrnlToLLVMPasses(pm, /*enableCSE=*/true, verifyInputTensors);
    }
  }

  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(*module)))
    return CompilerFailure;
  return emitOutput(module, context, outputNameNoExt,
      pm, emissionTarget, hardwareTarget);
}
} // namespace core_dnn
