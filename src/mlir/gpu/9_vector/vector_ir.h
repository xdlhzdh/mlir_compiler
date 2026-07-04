#pragma once
// vector_ir.h — P9 (9_vector): simplified Vector-dialect IR (with 8_scf_affine).
//
// Models the full journey from SCF scalar loops → vector dialect → LLVM IR:
//
//   VecOp   — individual vector operation (transfer_read/write, contract,
//             broadcast, fma, shape_cast, masking, llvm intrinsics, etc.)
//   VecFunc — function containing a flat list of VecOps (SSA-style)
//
// Each VecOp carries a human-readable annotation and the vector type
// information needed for register-level reasoning.
//
// Header-only, pure C++17, no external dependencies.

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace vec_ir {

// ======================== Element Type ========================

enum class ElemType { F32, F64, I32, I64 };

inline const char *elem_str(ElemType t) {
  switch (t) {
  case ElemType::F32: return "f32";
  case ElemType::F64: return "f64";
  case ElemType::I32: return "i32";
  case ElemType::I64: return "i64";
  }
  return "?";
}

// ======================== MemRef Type ========================

struct MemRefType {
  std::vector<int64_t> dims;
  ElemType elem = ElemType::F32;
  std::string str() const {
    std::ostringstream os;
    os << "memref<";
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i) os << "x";
      os << dims[i];
    }
    if (!dims.empty()) os << "x";
    os << elem_str(elem) << ">";
    return os.str();
  }
};

// ======================== Vector Type ========================

struct VecType {
  std::vector<int> shape;
  ElemType elem = ElemType::F32;
  std::string str() const {
    std::ostringstream os;
    os << "vector<";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i) os << "x";
      os << shape[i];
    }
    os << "x" << elem_str(elem) << ">";
    return os.str();
  }
  int total_lanes() const {
    int n = 1;
    for (int s : shape) n *= s;
    return n;
  }
};

// ======================== Op Kind ========================

enum class OpKind {
  COMMENT,
  LOOP,            // scf.for (wrapper for any remaining loops)
  TRANSFER_READ,   // vector.transfer_read %mem[idx] → %vec
  TRANSFER_WRITE,  // vector.transfer_write %vec, %mem[idx]
  BROADCAST,       // vector.broadcast %scalar → %vec
  SPLAT,           // vector.splat %scalar → %vec (constant fill)
  CONTRACT,        // vector.contract %A, %B, %acc → %result (matmul)
  FMA,             // vector.fma %a, %b, %c → %r
  ARITH,           // arith.addf / mulf / maxf / cmpf on vectors
  SHAPE_CAST,      // vector.shape_cast %src → %dst (reshape)
  EXTRACT_SLICE,   // vector.extract_strided_slice
  INSERT_SLICE,    // vector.insert_strided_slice
  MASK,            // vector.create_mask
  MASKED_LOAD,     // vector.maskedload
  MASKED_STORE,    // vector.maskedstore
  LLVM_INTRIN,     // llvm.intr.fmuladd / x86_avx2.fma / ...
  LLVM_LOAD,       // llvm.load (flat pointer)
  LLVM_STORE,      // llvm.store (flat pointer)
  ALLOCA,          // llvm.alloca (register spill / scratch)
};

// ======================== VecOp ========================

struct VecOp {
  OpKind kind = OpKind::COMMENT;

  std::string result;
  std::string arg0, arg1, arg2;
  std::string memref;
  std::vector<std::string> indices;
  VecType vtype;
  std::string annotation;

  // Loop-specific
  std::string iv;
  int64_t lb = 0, ub = 0, step = 1;
  std::vector<VecOp> body;

  // Mask-specific
  int mask_len = 0;
  int active_lanes = 0;

  // ─── Factory methods ───

  static VecOp Comment(const std::string &text) {
    VecOp o; o.kind = OpKind::COMMENT; o.annotation = text; return o;
  }

  static VecOp Loop(const std::string &iv_, int64_t lb_, int64_t ub_,
                    int64_t step_) {
    VecOp o; o.kind = OpKind::LOOP;
    o.iv = iv_; o.lb = lb_; o.ub = ub_; o.step = step_;
    return o;
  }

  static VecOp TransferRead(const std::string &res, const std::string &mem,
                            std::vector<std::string> idx, const VecType &vt,
                            const std::string &ann = "") {
    VecOp o; o.kind = OpKind::TRANSFER_READ;
    o.result = res; o.memref = mem; o.indices = std::move(idx);
    o.vtype = vt; o.annotation = ann;
    return o;
  }

  static VecOp TransferWrite(const std::string &val, const std::string &mem,
                             std::vector<std::string> idx, const VecType &vt,
                             const std::string &ann = "") {
    VecOp o; o.kind = OpKind::TRANSFER_WRITE;
    o.arg0 = val; o.memref = mem; o.indices = std::move(idx);
    o.vtype = vt; o.annotation = ann;
    return o;
  }

  static VecOp Broadcast(const std::string &res, const std::string &scalar,
                         const VecType &vt) {
    VecOp o; o.kind = OpKind::BROADCAST;
    o.result = res; o.arg0 = scalar; o.vtype = vt;
    return o;
  }

  static VecOp Splat(const std::string &res, const std::string &val,
                     const VecType &vt) {
    VecOp o; o.kind = OpKind::SPLAT;
    o.result = res; o.arg0 = val; o.vtype = vt;
    return o;
  }

  static VecOp Contract(const std::string &res, const std::string &a,
                        const std::string &b, const std::string &acc,
                        const VecType &vt, const std::string &ann = "") {
    VecOp o; o.kind = OpKind::CONTRACT;
    o.result = res; o.arg0 = a; o.arg1 = b; o.arg2 = acc; o.vtype = vt;
    o.annotation = ann;
    return o;
  }

  static VecOp FMA(const std::string &res, const std::string &a,
                   const std::string &b, const std::string &acc,
                   const VecType &vt) {
    VecOp o; o.kind = OpKind::FMA;
    o.result = res; o.arg0 = a; o.arg1 = b; o.arg2 = acc; o.vtype = vt;
    return o;
  }

  static VecOp Arith(const std::string &op_name, const std::string &res,
                     const std::string &a, const std::string &b,
                     const VecType &vt) {
    VecOp o; o.kind = OpKind::ARITH;
    o.annotation = op_name; o.result = res; o.arg0 = a; o.arg1 = b;
    o.vtype = vt;
    return o;
  }

  static VecOp ShapeCast(const std::string &res, const std::string &src,
                         const VecType &from, const VecType &to) {
    VecOp o; o.kind = OpKind::SHAPE_CAST;
    o.result = res; o.arg0 = src;
    o.annotation = from.str() + " → " + to.str();
    o.vtype = to;
    return o;
  }

  static VecOp ExtractSlice(const std::string &res, const std::string &src,
                            int offset, const VecType &vt) {
    VecOp o; o.kind = OpKind::EXTRACT_SLICE;
    o.result = res; o.arg0 = src;
    o.annotation = "offset=" + std::to_string(offset);
    o.vtype = vt;
    return o;
  }

  static VecOp InsertSlice(const std::string &res, const std::string &src,
                           const std::string &dst, int offset,
                           const VecType &vt) {
    VecOp o; o.kind = OpKind::INSERT_SLICE;
    o.result = res; o.arg0 = src; o.arg1 = dst;
    o.annotation = "offset=" + std::to_string(offset);
    o.vtype = vt;
    return o;
  }

  static VecOp CreateMask(const std::string &res, int len, int active) {
    VecOp o; o.kind = OpKind::MASK;
    o.result = res; o.mask_len = len; o.active_lanes = active;
    return o;
  }

  static VecOp MaskedLoad(const std::string &res, const std::string &mem,
                          std::vector<std::string> idx,
                          const std::string &mask, const VecType &vt) {
    VecOp o; o.kind = OpKind::MASKED_LOAD;
    o.result = res; o.memref = mem; o.indices = std::move(idx);
    o.arg0 = mask; o.vtype = vt;
    return o;
  }

  static VecOp MaskedStore(const std::string &val, const std::string &mem,
                           std::vector<std::string> idx,
                           const std::string &mask, const VecType &vt) {
    VecOp o; o.kind = OpKind::MASKED_STORE;
    o.arg0 = val; o.memref = mem; o.indices = std::move(idx);
    o.arg1 = mask; o.vtype = vt;
    return o;
  }

  static VecOp LLVMIntrin(const std::string &intrin, const std::string &res,
                          const std::string &a, const std::string &b,
                          const std::string &c, const VecType &vt) {
    VecOp o; o.kind = OpKind::LLVM_INTRIN;
    o.annotation = intrin; o.result = res;
    o.arg0 = a; o.arg1 = b; o.arg2 = c; o.vtype = vt;
    return o;
  }

  static VecOp LLVMLoad(const std::string &res, const std::string &ptr,
                        const VecType &vt) {
    VecOp o; o.kind = OpKind::LLVM_LOAD;
    o.result = res; o.arg0 = ptr; o.vtype = vt;
    return o;
  }

  static VecOp LLVMStore(const std::string &val, const std::string &ptr,
                         const VecType &vt) {
    VecOp o; o.kind = OpKind::LLVM_STORE;
    o.arg0 = val; o.arg1 = ptr; o.vtype = vt;
    return o;
  }

  static VecOp Alloca(const std::string &res, const VecType &vt) {
    VecOp o; o.kind = OpKind::ALLOCA;
    o.result = res; o.vtype = vt;
    return o;
  }

  // ─── Printing ───

  void print(std::ostream &os, int indent = 0) const {
    std::string pad(indent, ' ');
    switch (kind) {
    case OpKind::COMMENT:
      os << pad << "// " << annotation << "\n";
      break;
    case OpKind::LOOP:
      os << pad << "scf.for " << iv << " = " << lb << " to " << ub
         << " step " << step << " {\n";
      for (auto &s : body) s.print(os, indent + 2);
      os << pad << "}\n";
      break;
    case OpKind::TRANSFER_READ:
      os << pad << result << " = vector.transfer_read " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "] : " << vtype.str();
      if (!annotation.empty()) os << "  // " << annotation;
      os << "\n";
      break;
    case OpKind::TRANSFER_WRITE:
      os << pad << "vector.transfer_write " << arg0 << ", " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "] : " << vtype.str();
      if (!annotation.empty()) os << "  // " << annotation;
      os << "\n";
      break;
    case OpKind::BROADCAST:
      os << pad << result << " = vector.broadcast " << arg0
         << " : " << vtype.str() << "\n";
      break;
    case OpKind::SPLAT:
      os << pad << result << " = vector.splat " << arg0
         << " : " << vtype.str() << "\n";
      break;
    case OpKind::CONTRACT:
      os << pad << result << " = vector.contract " << arg0 << ", " << arg1
         << ", " << arg2 << " : " << vtype.str();
      if (!annotation.empty()) os << "  // " << annotation;
      os << "\n";
      break;
    case OpKind::FMA:
      os << pad << result << " = vector.fma " << arg0 << ", " << arg1
         << ", " << arg2 << " : " << vtype.str() << "\n";
      break;
    case OpKind::ARITH:
      os << pad << result << " = " << annotation << " " << arg0 << ", "
         << arg1 << " : " << vtype.str() << "\n";
      break;
    case OpKind::SHAPE_CAST:
      os << pad << result << " = vector.shape_cast " << arg0 << " : "
         << annotation << "\n";
      break;
    case OpKind::EXTRACT_SLICE:
      os << pad << result << " = vector.extract_strided_slice " << arg0
         << " {" << annotation << "} : " << vtype.str() << "\n";
      break;
    case OpKind::INSERT_SLICE:
      os << pad << result << " = vector.insert_strided_slice " << arg0
         << " into " << arg1 << " {" << annotation << "} : "
         << vtype.str() << "\n";
      break;
    case OpKind::MASK:
      os << pad << result << " = vector.create_mask " << active_lanes
         << " : vector<" << mask_len << "xi1>\n";
      break;
    case OpKind::MASKED_LOAD:
      os << pad << result << " = vector.maskedload " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "], " << arg0 << " : " << vtype.str() << "\n";
      break;
    case OpKind::MASKED_STORE:
      os << pad << "vector.maskedstore " << arg0 << ", " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "], " << arg1 << " : " << vtype.str() << "\n";
      break;
    case OpKind::LLVM_INTRIN:
      os << pad << result << " = " << annotation << " " << arg0 << ", "
         << arg1 << ", " << arg2 << " : " << vtype.str() << "\n";
      break;
    case OpKind::LLVM_LOAD:
      os << pad << result << " = llvm.load " << arg0
         << " : " << vtype.str() << "\n";
      break;
    case OpKind::LLVM_STORE:
      os << pad << "llvm.store " << arg0 << ", " << arg1
         << " : " << vtype.str() << "\n";
      break;
    case OpKind::ALLOCA:
      os << pad << result << " = llvm.alloca : " << vtype.str() << "\n";
      break;
    }
  }
};

// ======================== VecFunc ========================

struct VecFunc {
  std::string name;
  struct Arg {
    std::string name;
    MemRefType type;
  };
  std::vector<Arg> args;
  std::vector<VecOp> ops;

  void print(std::ostream &os) const {
    os << "func.func @" << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) os << ",\n    ";
      os << args[i].name << ": " << args[i].type.str();
    }
    os << ") {\n";
    for (auto &o : ops) o.print(os, 2);
    os << "  return\n}\n";
  }

  static int count_kind(const std::vector<VecOp> &ops, OpKind k) {
    int c = 0;
    for (auto &o : ops) {
      if (o.kind == k) ++c;
      if (o.kind == OpKind::LOOP) c += count_kind(o.body, k);
    }
    return c;
  }

  int count(OpKind k) const { return count_kind(ops, k); }
};

} // namespace vec_ir
