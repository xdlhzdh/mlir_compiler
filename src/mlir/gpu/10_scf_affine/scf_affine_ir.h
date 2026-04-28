#pragma once
// scf_affine_ir.h — Simplified SCF/Affine loop-level IR.
//
// Models the lowering from Linalg-on-memref to explicit loop nests:
//   • Stmt  — tree node: for-loop, parallel-loop, load, store, arith, vector, etc.
//   • Func  — function containing a flat list of statements (the program tree)
//
// Transformations demonstrated (9-stage pipeline):
//   Linalg → explicit scf.for/affine.for loops with memref.load/store
//   Loop canonicalization, tiling, interchange, fusion, parallelization
//   Memory promotion, register hoisting, vectorization preparation
//
// Header-only, pure C++17, no external dependencies.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace loop_ir {

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

inline int64_t elem_bytes(ElemType t) {
  switch (t) {
  case ElemType::F32: case ElemType::I32: return 4;
  case ElemType::F64: case ElemType::I64: return 8;
  }
  return 4;
}

// ======================== MemRef Type ========================

struct MemRefType {
  std::vector<int64_t> dims;
  ElemType elem = ElemType::F32;

  int64_t bytes() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n * elem_bytes(elem);
  }
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

// ======================== Statement Kind ========================

enum class StmtKind {
  FOR_LOOP,   // scf.for / affine.for
  PAR_LOOP,   // scf.parallel / affine.parallel
  LOAD,       // memref.load
  STORE,      // memref.store
  ARITH,      // arith.mulf, arith.addf, arith.maxf, vector.broadcast, etc.
  ALLOC,      // memref.alloc (stack promotion)
  DEALLOC,    // memref.dealloc
  COPY,       // memref.copy / linalg.copy (tile copy for promotion)
  VEC_LOAD,   // vector.load / vector.transfer_read
  VEC_STORE,  // vector.store / vector.transfer_write
  VEC_FMA,    // vector.fma / vector.contract
  COMMENT,    // annotation / comment
};

// ======================== Statement ========================

struct Stmt {
  StmtKind kind = StmtKind::COMMENT;

  // === Loop fields (FOR_LOOP / PAR_LOOP) ===
  std::string iv;
  int64_t lb = 0, ub = 0, step = 1;
  std::vector<Stmt> body;
  std::string thread_map;

  // === Load/Store fields ===
  std::string reg;
  std::string memref;
  std::vector<std::string> indices;

  // === Arith fields ===
  std::string op;
  std::string result, arg0, arg1;

  // === Alloc/Dealloc fields ===
  std::string alloc_name;
  MemRefType alloc_type;

  // === Copy fields ===
  std::string copy_src, copy_dst;

  // === Vector width (VEC_LOAD/VEC_STORE/VEC_FMA + broadcast ARITH) ===
  int vec_width = 0;

  // === Comment ===
  std::string comment;

  // ─── Factory methods ───

  static Stmt For(const std::string &iv_, int64_t lb_, int64_t ub_,
                  int64_t step_ = 1) {
    Stmt s;
    s.kind = StmtKind::FOR_LOOP;
    s.iv = iv_; s.lb = lb_; s.ub = ub_; s.step = step_;
    return s;
  }
  static Stmt ParFor(const std::string &iv_, int64_t lb_, int64_t ub_,
                     int64_t step_ = 1, const std::string &tmap = "") {
    Stmt s;
    s.kind = StmtKind::PAR_LOOP;
    s.iv = iv_; s.lb = lb_; s.ub = ub_; s.step = step_;
    s.thread_map = tmap;
    return s;
  }
  static Stmt Load(const std::string &reg_, const std::string &mem,
                   std::vector<std::string> idx) {
    Stmt s;
    s.kind = StmtKind::LOAD;
    s.reg = reg_; s.memref = mem; s.indices = std::move(idx);
    return s;
  }
  static Stmt Store(const std::string &reg_, const std::string &mem,
                    std::vector<std::string> idx) {
    Stmt s;
    s.kind = StmtKind::STORE;
    s.reg = reg_; s.memref = mem; s.indices = std::move(idx);
    return s;
  }
  static Stmt Arith(const std::string &op_, const std::string &res,
                    const std::string &a0, const std::string &a1 = "") {
    Stmt s;
    s.kind = StmtKind::ARITH;
    s.op = op_; s.result = res; s.arg0 = a0; s.arg1 = a1;
    return s;
  }
  static Stmt Alloc(const std::string &name, const MemRefType &t) {
    Stmt s;
    s.kind = StmtKind::ALLOC;
    s.alloc_name = name; s.alloc_type = t;
    return s;
  }
  static Stmt Dealloc(const std::string &name) {
    Stmt s;
    s.kind = StmtKind::DEALLOC;
    s.alloc_name = name;
    return s;
  }
  static Stmt CopyOp(const std::string &src, const std::string &dst) {
    Stmt s;
    s.kind = StmtKind::COPY;
    s.copy_src = src; s.copy_dst = dst;
    return s;
  }
  static Stmt VecLoad(const std::string &reg_, const std::string &mem,
                      std::vector<std::string> idx, int w) {
    Stmt s;
    s.kind = StmtKind::VEC_LOAD;
    s.reg = reg_; s.memref = mem; s.indices = std::move(idx); s.vec_width = w;
    return s;
  }
  static Stmt VecStore(const std::string &reg_, const std::string &mem,
                       std::vector<std::string> idx, int w) {
    Stmt s;
    s.kind = StmtKind::VEC_STORE;
    s.reg = reg_; s.memref = mem; s.indices = std::move(idx); s.vec_width = w;
    return s;
  }
  static Stmt VecFMA(const std::string &res, const std::string &a,
                     const std::string &b, const std::string &acc, int w) {
    Stmt s;
    s.kind = StmtKind::VEC_FMA;
    s.result = res; s.arg0 = a; s.arg1 = b;
    s.reg = acc;
    s.vec_width = w;
    return s;
  }
  static Stmt Comment(const std::string &text) {
    Stmt s;
    s.kind = StmtKind::COMMENT;
    s.comment = text;
    return s;
  }

  bool is_loop() const {
    return kind == StmtKind::FOR_LOOP || kind == StmtKind::PAR_LOOP;
  }

  // ─── Printing ───

  void print(std::ostream &os, int indent = 0) const {
    std::string pad(indent, ' ');
    switch (kind) {
    case StmtKind::FOR_LOOP:
      os << pad << "scf.for " << iv << " = " << lb << " to " << ub;
      if (step != 1) os << " step " << step;
      os << " {\n";
      for (auto &s : body) s.print(os, indent + 2);
      os << pad << "}\n";
      break;
    case StmtKind::PAR_LOOP:
      os << pad << "scf.parallel (" << iv << ") = (" << lb << ") to ("
         << ub << ")";
      if (step != 1) os << " step (" << step << ")";
      if (!thread_map.empty()) os << "  // " << thread_map;
      os << " {\n";
      for (auto &s : body) s.print(os, indent + 2);
      os << pad << "}\n";
      break;
    case StmtKind::LOAD:
      os << pad << reg << " = memref.load " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "]\n";
      break;
    case StmtKind::STORE:
      os << pad << "memref.store " << reg << ", " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "]\n";
      break;
    case StmtKind::ARITH:
      os << pad << result << " = " << op << " " << arg0;
      if (!arg1.empty()) os << ", " << arg1;
      if (vec_width > 0)
        os << " : vector<" << vec_width << "x" << elem_str(ElemType::F32)
           << ">";
      os << "\n";
      break;
    case StmtKind::ALLOC:
      os << pad << alloc_name << " = memref.alloc() : "
         << alloc_type.str() << "\n";
      break;
    case StmtKind::DEALLOC:
      os << pad << "memref.dealloc " << alloc_name << "\n";
      break;
    case StmtKind::COPY:
      os << pad << "memref.copy " << copy_src << " -> " << copy_dst << "\n";
      break;
    case StmtKind::VEC_LOAD:
      os << pad << reg << " = vector.load " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "] : vector<" << vec_width << "x" << elem_str(ElemType::F32)
         << ">\n";
      break;
    case StmtKind::VEC_STORE:
      os << pad << "vector.store " << reg << ", " << memref << "[";
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i) os << ", ";
        os << indices[i];
      }
      os << "] : vector<" << vec_width << "x" << elem_str(ElemType::F32)
         << ">\n";
      break;
    case StmtKind::VEC_FMA:
      os << pad << result << " = vector.fma " << arg0 << ", " << arg1
         << ", " << reg << " : vector<" << vec_width << "x"
         << elem_str(ElemType::F32) << ">\n";
      break;
    case StmtKind::COMMENT:
      os << pad << "// " << comment << "\n";
      break;
    }
  }
};

// ======================== Function ========================

struct Func {
  std::string name;
  struct Arg {
    std::string name;
    MemRefType type;
  };
  std::vector<Arg> args;
  std::vector<Stmt> body;

  void print(std::ostream &os) const {
    os << "func.func @" << name << "(";
    for (size_t i = 0; i < args.size(); ++i) {
      if (i) os << ",\n    ";
      os << args[i].name << ": " << args[i].type.str();
    }
    os << ") {\n";
    for (auto &s : body) s.print(os, 2);
    os << "  return\n}\n";
  }

  static int count_loops(const std::vector<Stmt> &stmts) {
    int c = 0;
    for (auto &s : stmts) {
      if (s.is_loop()) {
        ++c;
        c += count_loops(s.body);
      }
    }
    return c;
  }
  static int count_ops(const std::vector<Stmt> &stmts) {
    int c = 0;
    for (auto &s : stmts) {
      if (!s.is_loop() && s.kind != StmtKind::COMMENT) ++c;
      if (s.is_loop()) c += count_ops(s.body);
    }
    return c;
  }
  static int count_par(const std::vector<Stmt> &stmts) {
    int c = 0;
    for (auto &s : stmts) {
      if (s.kind == StmtKind::PAR_LOOP) ++c;
      if (s.is_loop()) c += count_par(s.body);
    }
    return c;
  }
  static int count_vec(const std::vector<Stmt> &stmts) {
    int c = 0;
    for (auto &s : stmts) {
      if (s.kind == StmtKind::VEC_LOAD || s.kind == StmtKind::VEC_STORE ||
          s.kind == StmtKind::VEC_FMA)
        ++c;
      if (s.is_loop()) c += count_vec(s.body);
    }
    return c;
  }

  int total_loops() const { return count_loops(body); }
  int total_ops() const { return count_ops(body); }
  int total_par() const { return count_par(body); }
  int total_vec() const { return count_vec(body); }
};

} // namespace loop_ir
