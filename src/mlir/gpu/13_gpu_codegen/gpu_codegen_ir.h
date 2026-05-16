#pragma once
// gpu_codegen_ir.h — P10 (13_gpu_codegen): simplified IR for GPU code generation (concept).
//
// Models the full journey from high-level parallel ops → GPU dialect → NVVM/PTX:
//
//   ParallelOp    — scf.parallel / linalg.generic with parallel iterators
//   GPUOp         — gpu.launch_func, gpu.block_id, gpu.thread_id, etc.
//   NVVMOp        — NVVM dialect ops (thread.idx, ctaid, barrier, shared memory)
//   PTXInstr      — PTX assembly instructions
//   KernelConfig  — launch configuration (grid, block, shared memory)
//
// Header-only, pure C++17, no external dependencies.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace gpu_ir {

// ======================== Element Type ========================

enum class ElemType { F16, F32, F64, I32, I64 };

inline const char *elem_str(ElemType t) {
  switch (t) {
  case ElemType::F16: return "f16";
  case ElemType::F32: return "f32";
  case ElemType::F64: return "f64";
  case ElemType::I32: return "i32";
  case ElemType::I64: return "i64";
  }
  return "?";
}

inline int elem_bytes(ElemType t) {
  switch (t) {
  case ElemType::F16: return 2;
  case ElemType::F32: return 4;
  case ElemType::F64: return 8;
  case ElemType::I32: return 4;
  case ElemType::I64: return 8;
  }
  return 4;
}

// ======================== MemRef Type ========================

struct MemRefType {
  std::vector<int64_t> dims;
  ElemType elem = ElemType::F32;
  enum Space { GLOBAL = 0, SHARED = 3, LOCAL = 5 };
  Space space = GLOBAL;

  std::string str() const {
    std::ostringstream os;
    os << "memref<";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i) os << "x";
      os << dims[i];
    }
    if (!dims.empty()) os << "x";
    os << elem_str(elem);
    if (space == SHARED) os << ", #gpu.address_space<workgroup>";
    else if (space == LOCAL) os << ", #gpu.address_space<private>";
    os << ">";
    return os.str();
  }

  int64_t numel() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
  }
  int64_t bytes() const { return numel() * elem_bytes(elem); }
};

// ======================== Kernel Launch Config ========================

struct Dim3 {
  int x = 1, y = 1, z = 1;
  std::string str() const {
    std::ostringstream os;
    os << "(" << x << ", " << y << ", " << z << ")";
    return os.str();
  }
  int total() const { return x * y * z; }
};

struct KernelConfig {
  std::string name;
  Dim3 grid;
  Dim3 block;
  int shared_mem_bytes = 0;
  int reg_per_thread = 32;

  void print(std::ostream &os) const {
    os << "  gpu.launch_func @" << name << "\n"
       << "    gridDim  = " << grid.str() << "\n"
       << "    blockDim = " << block.str() << "\n"
       << "    shared   = " << shared_mem_bytes << " bytes\n"
       << "    regs     = " << reg_per_thread << " per thread\n";
  }

  int total_threads() const { return grid.total() * block.total(); }

  double occupancy_estimate(int sm_max_threads = 2048,
                            int sm_max_blocks = 32,
                            int sm_max_regs = 65536,
                            int sm_max_smem = 49152) const {
    int blocks_by_threads = sm_max_threads / block.total();
    int blocks_by_regs =
        sm_max_regs / (block.total() * reg_per_thread);
    int blocks_by_smem =
        shared_mem_bytes > 0 ? sm_max_smem / shared_mem_bytes : sm_max_blocks;
    int blocks_per_sm =
        std::min({blocks_by_threads, blocks_by_regs, blocks_by_smem,
                  sm_max_blocks});
    if (blocks_per_sm <= 0) return 0.0;
    double active = blocks_per_sm * block.total();
    return active / sm_max_threads;
  }
};

// ======================== GPU Dialect Op ========================

enum class GPUOpKind {
  COMMENT,
  MODULE_BEGIN,
  MODULE_END,
  FUNC_BEGIN,
  FUNC_END,
  BLOCK_ID_X, BLOCK_ID_Y, BLOCK_ID_Z,
  THREAD_ID_X, THREAD_ID_Y, THREAD_ID_Z,
  BLOCK_DIM_X,
  GRID_DIM_X,
  GLOBAL_LOAD,
  GLOBAL_STORE,
  SHARED_LOAD,
  SHARED_STORE,
  BARRIER,
  ARITH_ADDF,
  ARITH_MULF,
  ARITH_MAXF,
  ARITH_ADDI,
  ARITH_MULI,
  INDEX_CAST,
  SCF_FOR_BEGIN,
  SCF_FOR_END,
  ALLOC_SHARED,
};

struct GPUOp {
  GPUOpKind kind = GPUOpKind::COMMENT;
  std::string dest;
  std::vector<std::string> srcs;
  std::string annotation;
  ElemType ty = ElemType::F32;

  std::string mnemonic() const {
    switch (kind) {
    case GPUOpKind::COMMENT:       return "//";
    case GPUOpKind::MODULE_BEGIN:  return "gpu.module";
    case GPUOpKind::MODULE_END:    return "}";
    case GPUOpKind::FUNC_BEGIN:    return "gpu.func";
    case GPUOpKind::FUNC_END:      return "}";
    case GPUOpKind::BLOCK_ID_X:    return "gpu.block_id x";
    case GPUOpKind::BLOCK_ID_Y:    return "gpu.block_id y";
    case GPUOpKind::BLOCK_ID_Z:    return "gpu.block_id z";
    case GPUOpKind::THREAD_ID_X:   return "gpu.thread_id x";
    case GPUOpKind::THREAD_ID_Y:   return "gpu.thread_id y";
    case GPUOpKind::THREAD_ID_Z:   return "gpu.thread_id z";
    case GPUOpKind::BLOCK_DIM_X:   return "gpu.block_dim x";
    case GPUOpKind::GRID_DIM_X:    return "gpu.grid_dim x";
    case GPUOpKind::GLOBAL_LOAD:   return "memref.load";
    case GPUOpKind::GLOBAL_STORE:  return "memref.store";
    case GPUOpKind::SHARED_LOAD:   return "memref.load /*shared*/";
    case GPUOpKind::SHARED_STORE:  return "memref.store /*shared*/";
    case GPUOpKind::BARRIER:       return "gpu.barrier";
    case GPUOpKind::ARITH_ADDF:    return "arith.addf";
    case GPUOpKind::ARITH_MULF:    return "arith.mulf";
    case GPUOpKind::ARITH_MAXF:    return "arith.maxf";
    case GPUOpKind::ARITH_ADDI:    return "arith.addi";
    case GPUOpKind::ARITH_MULI:    return "arith.muli";
    case GPUOpKind::INDEX_CAST:    return "arith.index_cast";
    case GPUOpKind::SCF_FOR_BEGIN: return "scf.for";
    case GPUOpKind::SCF_FOR_END:   return "}";
    case GPUOpKind::ALLOC_SHARED:  return "memref.alloc /*workgroup*/";
    }
    return "?";
  }
};

// ======================== NVVM Dialect Op ========================

enum class NVVMOpKind {
  COMMENT,
  FUNC_BEGIN,
  FUNC_END,
  TID_X, TID_Y, TID_Z,
  CTAID_X, CTAID_Y, CTAID_Z,
  NTID_X,
  NCTAID_X,
  SHARED_ALLOC,
  LD_GLOBAL,
  ST_GLOBAL,
  LD_SHARED,
  ST_SHARED,
  BAR_SYNC,
  FMA_F32,
  ADD_F32,
  MUL_F32,
  MAX_F32,
  ADD_I32,
  MUL_I32,
  SHL_I32,
  CVT_U32_U16,
  BR,
  BR_COND,
  ICMP,
  PHI,
  RET,
};

struct NVVMOp {
  NVVMOpKind kind = NVVMOpKind::COMMENT;
  std::string dest;
  std::vector<std::string> srcs;
  std::string annotation;

  std::string mnemonic() const {
    switch (kind) {
    case NVVMOpKind::COMMENT:      return "//";
    case NVVMOpKind::FUNC_BEGIN:   return "llvm.func @kernel";
    case NVVMOpKind::FUNC_END:     return "}";
    case NVVMOpKind::TID_X:        return "nvvm.read.ptx.sreg.tid.x";
    case NVVMOpKind::TID_Y:        return "nvvm.read.ptx.sreg.tid.y";
    case NVVMOpKind::TID_Z:        return "nvvm.read.ptx.sreg.tid.z";
    case NVVMOpKind::CTAID_X:      return "nvvm.read.ptx.sreg.ctaid.x";
    case NVVMOpKind::CTAID_Y:      return "nvvm.read.ptx.sreg.ctaid.y";
    case NVVMOpKind::CTAID_Z:      return "nvvm.read.ptx.sreg.ctaid.z";
    case NVVMOpKind::NTID_X:       return "nvvm.read.ptx.sreg.ntid.x";
    case NVVMOpKind::NCTAID_X:     return "nvvm.read.ptx.sreg.nctaid.x";
    case NVVMOpKind::SHARED_ALLOC: return "llvm.mlir.addressof @shared";
    case NVVMOpKind::LD_GLOBAL:    return "llvm.load /*global*/";
    case NVVMOpKind::ST_GLOBAL:    return "llvm.store /*global*/";
    case NVVMOpKind::LD_SHARED:    return "llvm.load /*shared*/";
    case NVVMOpKind::ST_SHARED:    return "llvm.store /*shared*/";
    case NVVMOpKind::BAR_SYNC:     return "nvvm.barrier0";
    case NVVMOpKind::FMA_F32:      return "llvm.intr.fma";
    case NVVMOpKind::ADD_F32:      return "llvm.fadd";
    case NVVMOpKind::MUL_F32:      return "llvm.fmul";
    case NVVMOpKind::MAX_F32:      return "llvm.intr.maxnum";
    case NVVMOpKind::ADD_I32:      return "llvm.add";
    case NVVMOpKind::MUL_I32:      return "llvm.mul";
    case NVVMOpKind::SHL_I32:      return "llvm.shl";
    case NVVMOpKind::CVT_U32_U16:  return "llvm.zext";
    case NVVMOpKind::BR:           return "llvm.br";
    case NVVMOpKind::BR_COND:      return "llvm.cond_br";
    case NVVMOpKind::ICMP:         return "llvm.icmp";
    case NVVMOpKind::PHI:          return "llvm.phi";
    case NVVMOpKind::RET:          return "llvm.return";
    }
    return "?";
  }
};

// ======================== PTX Instruction ========================

struct PTXInstr {
  std::string mnemonic;
  std::vector<std::string> operands;
  std::string annotation;

  std::string str() const {
    std::ostringstream os;
    os << "    " << std::left << std::setw(24) << mnemonic;
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i) os << ", ";
      os << operands[i];
    }
    if (!annotation.empty()) os << ";  // " << annotation;
    else os << ";";
    return os.str();
  }
};

// ======================== Tiling Strategy ========================

struct TileStrategy {
  std::string op_name;
  std::vector<int> tile_sizes;
  int block_x = 1, block_y = 1;
  bool use_shared_mem = false;
  int shared_tile_m = 0, shared_tile_n = 0, shared_tile_k = 0;

  void print(std::ostream &os) const {
    os << "  op: " << op_name << "\n"
       << "  tile = [";
    for (size_t i = 0; i < tile_sizes.size(); ++i) {
      if (i) os << ", ";
      os << tile_sizes[i];
    }
    os << "]\n"
       << "  block = (" << block_x << ", " << block_y << ")\n";
    if (use_shared_mem)
      os << "  shared_tile = [" << shared_tile_m << ", " << shared_tile_n
         << ", " << shared_tile_k << "]\n";
  }
};

}  // namespace gpu_ir
