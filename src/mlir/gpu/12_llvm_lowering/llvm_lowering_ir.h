#pragma once
// llvm_lowering_ir.h — Multi-level IR for the Vector → Machine Code pipeline.
//
// Models the full backend journey:
//
//   VecOp (from vector_ir.h)      — vector dialect input
//   LLVMOp                        — LLVM dialect / LLVM IR (SSA, typed)
//   MachineInstr                  — target-specific instruction (after ISel)
//   RegAlloc result               — virtual→physical mapping + spills
//   SchedSlot                     — scheduled instructions with cycle info
//   AsmLine                       — final assembly text + optional encoding
//
// Header-only, pure C++17, no external dependencies.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace backend_ir {

// ======================== LLVM Type ========================

enum class LLVMTy { VOID, I1, I32, I64, F32, F64, VEC8F32, VEC4F32, PTR };

inline const char *llty_str(LLVMTy t) {
  switch (t) {
  case LLVMTy::VOID:    return "void";
  case LLVMTy::I1:      return "i1";
  case LLVMTy::I32:     return "i32";
  case LLVMTy::I64:     return "i64";
  case LLVMTy::F32:     return "float";
  case LLVMTy::F64:     return "double";
  case LLVMTy::VEC8F32: return "<8 x float>";
  case LLVMTy::VEC4F32: return "<4 x float>";
  case LLVMTy::PTR:     return "ptr";
  }
  return "?";
}

// ======================== LLVM Op (Stage 1-2) ========================

enum class LLVMOpKind {
  COMMENT,
  FUNC_BEGIN,
  FUNC_END,
  LABEL,
  // Memory
  GEP,          // getelementptr
  LOAD,         // load <ty>, ptr
  STORE,        // store <ty> %val, ptr
  ALLOCA,       // alloca
  // Arithmetic
  FADD,         // fadd <8 x float>
  FMUL,         // fmul <8 x float>
  FSUB,
  // Intrinsics
  FMA,          // call @llvm.fma.v8f32
  MAXNUM,       // call @llvm.maxnum.v8f32
  // Vector
  BROADCAST,    // shufflevector (splat)
  EXTRACTELT,   // extractelement
  INSERTELT,    // insertelement
  // Control flow
  BR,           // br label
  BR_COND,      // br i1, label, label
  ICMP,         // icmp slt i64
  PHI,          // phi
  ADD_I64,      // add i64 (loop counter)
  RET,          // ret void
};

struct LLVMOp {
  LLVMOpKind kind = LLVMOpKind::COMMENT;
  std::string result;
  LLVMTy rtype = LLVMTy::VOID;
  std::string op0, op1, op2;
  std::string annotation;

  // GEP-specific
  std::string base_ptr;
  std::string index_expr;

  // Branch
  std::string label_true, label_false;

  // PHI
  std::vector<std::pair<std::string, std::string>> phi_pairs;

  static LLVMOp Comment(const std::string &t) {
    LLVMOp o; o.kind = LLVMOpKind::COMMENT; o.annotation = t; return o;
  }
  static LLVMOp FuncBegin(const std::string &name) {
    LLVMOp o; o.kind = LLVMOpKind::FUNC_BEGIN; o.result = name; return o;
  }
  static LLVMOp FuncEnd() {
    LLVMOp o; o.kind = LLVMOpKind::FUNC_END; return o;
  }
  static LLVMOp Label(const std::string &l) {
    LLVMOp o; o.kind = LLVMOpKind::LABEL; o.result = l; return o;
  }
  static LLVMOp GEP(const std::string &res, const std::string &base,
                     const std::string &idx, LLVMTy ety = LLVMTy::F32) {
    LLVMOp o; o.kind = LLVMOpKind::GEP;
    o.result = res; o.base_ptr = base; o.index_expr = idx; o.rtype = LLVMTy::PTR;
    o.annotation = llty_str(ety);
    return o;
  }
  static LLVMOp Load(const std::string &res, const std::string &ptr,
                      LLVMTy ty, int align = 0) {
    LLVMOp o; o.kind = LLVMOpKind::LOAD;
    o.result = res; o.op0 = ptr; o.rtype = ty;
    if (align > 0) o.annotation = "align " + std::to_string(align);
    return o;
  }
  static LLVMOp Store(const std::string &val, const std::string &ptr,
                       LLVMTy ty, int align = 0) {
    LLVMOp o; o.kind = LLVMOpKind::STORE;
    o.op0 = val; o.op1 = ptr; o.rtype = ty;
    if (align > 0) o.annotation = "align " + std::to_string(align);
    return o;
  }
  static LLVMOp FAdd(const std::string &res, const std::string &a,
                      const std::string &b, LLVMTy ty = LLVMTy::VEC8F32) {
    LLVMOp o; o.kind = LLVMOpKind::FADD;
    o.result = res; o.op0 = a; o.op1 = b; o.rtype = ty; return o;
  }
  static LLVMOp FMul(const std::string &res, const std::string &a,
                      const std::string &b, LLVMTy ty = LLVMTy::VEC8F32) {
    LLVMOp o; o.kind = LLVMOpKind::FMUL;
    o.result = res; o.op0 = a; o.op1 = b; o.rtype = ty; return o;
  }
  static LLVMOp FMA(const std::string &res, const std::string &a,
                     const std::string &b, const std::string &c,
                     LLVMTy ty = LLVMTy::VEC8F32) {
    LLVMOp o; o.kind = LLVMOpKind::FMA;
    o.result = res; o.op0 = a; o.op1 = b; o.op2 = c; o.rtype = ty; return o;
  }
  static LLVMOp MaxNum(const std::string &res, const std::string &a,
                        const std::string &b, LLVMTy ty = LLVMTy::VEC8F32) {
    LLVMOp o; o.kind = LLVMOpKind::MAXNUM;
    o.result = res; o.op0 = a; o.op1 = b; o.rtype = ty; return o;
  }
  static LLVMOp Broadcast(const std::string &res, const std::string &scalar,
                           LLVMTy ty = LLVMTy::VEC8F32) {
    LLVMOp o; o.kind = LLVMOpKind::BROADCAST;
    o.result = res; o.op0 = scalar; o.rtype = ty; return o;
  }
  static LLVMOp Br(const std::string &label) {
    LLVMOp o; o.kind = LLVMOpKind::BR; o.label_true = label; return o;
  }
  static LLVMOp BrCond(const std::string &cond, const std::string &t,
                         const std::string &f) {
    LLVMOp o; o.kind = LLVMOpKind::BR_COND;
    o.op0 = cond; o.label_true = t; o.label_false = f; return o;
  }
  static LLVMOp ICmp(const std::string &res, const std::string &pred,
                      const std::string &a, const std::string &b) {
    LLVMOp o; o.kind = LLVMOpKind::ICMP;
    o.result = res; o.annotation = pred; o.op0 = a; o.op1 = b;
    o.rtype = LLVMTy::I1;
    return o;
  }
  static LLVMOp Phi(const std::string &res, LLVMTy ty,
                     std::vector<std::pair<std::string, std::string>> pairs) {
    LLVMOp o; o.kind = LLVMOpKind::PHI;
    o.result = res; o.rtype = ty; o.phi_pairs = std::move(pairs); return o;
  }
  static LLVMOp AddI64(const std::string &res, const std::string &a,
                         const std::string &b) {
    LLVMOp o; o.kind = LLVMOpKind::ADD_I64;
    o.result = res; o.op0 = a; o.op1 = b; o.rtype = LLVMTy::I64; return o;
  }
  static LLVMOp Ret() {
    LLVMOp o; o.kind = LLVMOpKind::RET; return o;
  }

  void print(std::ostream &os, int indent = 2) const {
    std::string pad(indent, ' ');
    switch (kind) {
    case LLVMOpKind::COMMENT:
      os << pad << "; " << annotation << "\n"; break;
    case LLVMOpKind::FUNC_BEGIN:
      os << "define void @" << result << "(ptr %A, ptr %B, ptr %bias, ptr %C) {\n";
      break;
    case LLVMOpKind::FUNC_END:
      os << "}\n"; break;
    case LLVMOpKind::LABEL:
      os << result << ":\n"; break;
    case LLVMOpKind::GEP:
      os << pad << result << " = getelementptr " << annotation << ", ptr "
         << base_ptr << ", i64 " << index_expr << "\n"; break;
    case LLVMOpKind::LOAD:
      os << pad << result << " = load " << llty_str(rtype) << ", ptr " << op0;
      if (!annotation.empty()) os << ", " << annotation;
      os << "\n"; break;
    case LLVMOpKind::STORE:
      os << pad << "store " << llty_str(rtype) << " " << op0 << ", ptr " << op1;
      if (!annotation.empty()) os << ", " << annotation;
      os << "\n"; break;
    case LLVMOpKind::FADD:
      os << pad << result << " = fadd " << llty_str(rtype) << " " << op0
         << ", " << op1 << "\n"; break;
    case LLVMOpKind::FMUL:
      os << pad << result << " = fmul " << llty_str(rtype) << " " << op0
         << ", " << op1 << "\n"; break;
    case LLVMOpKind::FSUB:
      os << pad << result << " = fsub " << llty_str(rtype) << " " << op0
         << ", " << op1 << "\n"; break;
    case LLVMOpKind::FMA:
      os << pad << result << " = call " << llty_str(rtype)
         << " @llvm.fma.v8f32(" << llty_str(rtype) << " " << op0 << ", "
         << llty_str(rtype) << " " << op1 << ", "
         << llty_str(rtype) << " " << op2 << ")\n"; break;
    case LLVMOpKind::MAXNUM:
      os << pad << result << " = call " << llty_str(rtype)
         << " @llvm.maxnum.v8f32(" << llty_str(rtype) << " " << op0 << ", "
         << llty_str(rtype) << " " << op1 << ")\n"; break;
    case LLVMOpKind::BROADCAST:
      os << pad << result << " = shufflevector <1 x float> " << op0
         << ", <1 x float> poison, <8 x i32> zeroinitializer"
         << "  ; splat → " << llty_str(rtype) << "\n"; break;
    case LLVMOpKind::BR:
      os << pad << "br label %" << label_true << "\n"; break;
    case LLVMOpKind::BR_COND:
      os << pad << "br i1 " << op0 << ", label %" << label_true
         << ", label %" << label_false << "\n"; break;
    case LLVMOpKind::ICMP:
      os << pad << result << " = icmp " << annotation << " i64 " << op0
         << ", " << op1 << "\n"; break;
    case LLVMOpKind::PHI:
      os << pad << result << " = phi " << llty_str(rtype) << " ";
      for (size_t i = 0; i < phi_pairs.size(); ++i) {
        if (i) os << ", ";
        os << "[ " << phi_pairs[i].first << ", %" << phi_pairs[i].second << " ]";
      }
      os << "\n"; break;
    case LLVMOpKind::ADD_I64:
      os << pad << result << " = add i64 " << op0 << ", " << op1 << "\n"; break;
    case LLVMOpKind::RET:
      os << pad << "ret void\n"; break;
    default: break;
    }
  }
};

struct LLVMFunc {
  std::vector<LLVMOp> ops;
  void print(std::ostream &os) const {
    for (auto &o : ops) o.print(os);
  }
  int count(LLVMOpKind k) const {
    int c = 0;
    for (auto &o : ops) if (o.kind == k) ++c;
    return c;
  }
};

// ======================== Machine Instruction (Stage 3) ========================

enum class MIOpKind {
  COMMENT,
  LABEL,
  // x86 AVX2
  VMOVAPS_LOAD,     // vmovaps ymm, [mem]
  VMOVAPS_STORE,    // vmovaps [mem], ymm
  VBROADCASTSS,     // vbroadcastss ymm, [mem]
  VFMADD231PS,      // vfmadd231ps ymm, ymm, ymm
  VADDPS,           // vaddps ymm, ymm, ymm
  VMAXPS,           // vmaxps ymm, ymm, ymm
  VXORPS,           // vxorps ymm, ymm, ymm (zero reg)
  // Scalar / address
  LEA,              // lea rax, [base + idx*scale + disp]
  MOV,              // mov reg, imm / reg
  ADD,              // add reg, imm
  CMP,              // cmp reg, imm
  JL,               // jl label
  JMP,              // jmp label
  RET,
};

inline const char *mi_mnemonic(MIOpKind k) {
  switch (k) {
  case MIOpKind::COMMENT:       return ";";
  case MIOpKind::LABEL:         return "";
  case MIOpKind::VMOVAPS_LOAD:  return "vmovaps";
  case MIOpKind::VMOVAPS_STORE: return "vmovaps";
  case MIOpKind::VBROADCASTSS:  return "vbroadcastss";
  case MIOpKind::VFMADD231PS:   return "vfmadd231ps";
  case MIOpKind::VADDPS:        return "vaddps";
  case MIOpKind::VMAXPS:        return "vmaxps";
  case MIOpKind::VXORPS:        return "vxorps";
  case MIOpKind::LEA:           return "lea";
  case MIOpKind::MOV:           return "mov";
  case MIOpKind::ADD:           return "add";
  case MIOpKind::CMP:           return "cmp";
  case MIOpKind::JL:            return "jl";
  case MIOpKind::JMP:           return "jmp";
  case MIOpKind::RET:           return "ret";
  }
  return "???";
}

struct MachineInstr {
  MIOpKind kind = MIOpKind::COMMENT;
  std::string dst;
  std::string src1, src2, src3;
  std::string annotation;
  int latency = 1;
  int id = 0;
  std::vector<int> deps;

  static MachineInstr Comment(const std::string &t) {
    MachineInstr m; m.kind = MIOpKind::COMMENT; m.annotation = t; return m;
  }
  static MachineInstr LabelM(const std::string &l) {
    MachineInstr m; m.kind = MIOpKind::LABEL; m.dst = l; return m;
  }
  static MachineInstr VMovapsLoad(const std::string &dst, const std::string &mem) {
    MachineInstr m; m.kind = MIOpKind::VMOVAPS_LOAD;
    m.dst = dst; m.src1 = mem; m.latency = 5; return m;
  }
  static MachineInstr VMovapsStore(const std::string &mem, const std::string &src) {
    MachineInstr m; m.kind = MIOpKind::VMOVAPS_STORE;
    m.dst = mem; m.src1 = src; m.latency = 5; return m;
  }
  static MachineInstr VBroadcastss(const std::string &dst, const std::string &mem) {
    MachineInstr m; m.kind = MIOpKind::VBROADCASTSS;
    m.dst = dst; m.src1 = mem; m.latency = 5; return m;
  }
  static MachineInstr VFmadd231ps(const std::string &dst, const std::string &s1,
                                   const std::string &s2) {
    MachineInstr m; m.kind = MIOpKind::VFMADD231PS;
    m.dst = dst; m.src1 = s1; m.src2 = s2; m.latency = 4; return m;
  }
  static MachineInstr VAddps(const std::string &dst, const std::string &s1,
                              const std::string &s2) {
    MachineInstr m; m.kind = MIOpKind::VADDPS;
    m.dst = dst; m.src1 = s1; m.src2 = s2; m.latency = 4; return m;
  }
  static MachineInstr VMaxps(const std::string &dst, const std::string &s1,
                              const std::string &s2) {
    MachineInstr m; m.kind = MIOpKind::VMAXPS;
    m.dst = dst; m.src1 = s1; m.src2 = s2; m.latency = 4; return m;
  }
  static MachineInstr VXorps(const std::string &dst, const std::string &s1,
                              const std::string &s2) {
    MachineInstr m; m.kind = MIOpKind::VXORPS;
    m.dst = dst; m.src1 = s1; m.src2 = s2; m.latency = 1; return m;
  }
  static MachineInstr Lea(const std::string &dst, const std::string &addr) {
    MachineInstr m; m.kind = MIOpKind::LEA;
    m.dst = dst; m.src1 = addr; m.latency = 1; return m;
  }
  static MachineInstr Mov(const std::string &dst, const std::string &src) {
    MachineInstr m; m.kind = MIOpKind::MOV;
    m.dst = dst; m.src1 = src; m.latency = 1; return m;
  }
  static MachineInstr AddI(const std::string &dst, const std::string &imm) {
    MachineInstr m; m.kind = MIOpKind::ADD;
    m.dst = dst; m.src1 = imm; m.latency = 1; return m;
  }
  static MachineInstr Cmp(const std::string &reg, const std::string &imm) {
    MachineInstr m; m.kind = MIOpKind::CMP;
    m.dst = reg; m.src1 = imm; m.latency = 1; return m;
  }
  static MachineInstr Jl(const std::string &label) {
    MachineInstr m; m.kind = MIOpKind::JL;
    m.dst = label; m.latency = 1; return m;
  }
  static MachineInstr Jmp(const std::string &label) {
    MachineInstr m; m.kind = MIOpKind::JMP;
    m.dst = label; m.latency = 1; return m;
  }
  static MachineInstr RetM() {
    MachineInstr m; m.kind = MIOpKind::RET; m.latency = 1; return m;
  }

  void print(std::ostream &os, int indent = 4) const {
    std::string pad(indent, ' ');
    if (kind == MIOpKind::COMMENT) {
      os << pad << "; " << annotation << "\n"; return;
    }
    if (kind == MIOpKind::LABEL) {
      os << dst << ":\n"; return;
    }
    if (kind == MIOpKind::RET) {
      os << pad << "ret\n"; return;
    }
    os << pad << mi_mnemonic(kind);
    if (!dst.empty()) os << " " << dst;
    if (!src1.empty()) os << ", " << src1;
    if (!src2.empty()) os << ", " << src2;
    if (!src3.empty()) os << ", " << src3;
    if (!annotation.empty()) os << "    ; " << annotation;
    os << "\n";
  }
};

// ======================== Register Allocation (Stage 4) ========================

struct RegMapping {
  std::string vreg;
  std::string phys;
  bool spilled = false;
  int spill_slot = -1;

  void print(std::ostream &os) const {
    if (spilled) {
      std::printf("    %-12s → [stack+%d]  (SPILLED)\n",
                  vreg.c_str(), spill_slot);
    } else {
      std::printf("    %-12s → %-8s\n", vreg.c_str(), phys.c_str());
    }
  }
};

// ======================== Scheduling (Stage 5) ========================

struct SchedSlot {
  int cycle = 0;
  int instr_id = 0;
  std::string mnemonic;
  std::string operands;
  int port = 0;
  std::string annotation;

  void print(std::ostream &os) const {
    std::printf("    cycle %2d │ port %d │ %-14s %-30s",
                cycle, port, mnemonic.c_str(), operands.c_str());
    if (!annotation.empty()) std::printf("  ; %s", annotation.c_str());
    std::printf("\n");
  }
};

// ======================== Assembly Line (Stage 6) ========================

struct AsmLine {
  std::string label;
  std::string mnemonic;
  std::string operands;
  std::string encoding;  // hex bytes
  std::string comment;
  int offset = -1;

  void print(std::ostream &os) const {
    if (!label.empty()) {
      os << label << ":\n";
      return;
    }
    if (offset >= 0) {
      std::printf("  %04x: %-24s", offset, encoding.c_str());
    } else {
      os << "        ";
      std::printf("%-24s", "");
    }
    os << mnemonic;
    if (!operands.empty()) os << " " << operands;
    if (!comment.empty()) os << "    ; " << comment;
    os << "\n";
  }
};

} // namespace backend_ir
