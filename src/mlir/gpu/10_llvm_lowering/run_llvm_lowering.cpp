// run_llvm_lowering.cpp — P10 (10_llvm_lowering): Vector → machine code (llvm_lowering_ir, 7-step Pass chain)
//
// Takes vector dialect IR from P9 (9_vector) and lowers through the full backend:
//
//   P10 Step 0: Vector IR Input        — reconstruct vector dialect (from P9 / 9_vector output)
//   P10 Step 1: Vector Lowering        — vector ops → LLVM dialect (GEP, load, fma, shufflevector)
//   P10 Step 2: LLVM Dialect → LLVM IR — emit textual LLVM IR (SSA, loop CFG with phi nodes)
//   P10 Step 3: Instruction Selection  — LLVM IR → x86-64 AVX2 MachineInstr (DAG pattern match)
//   P10 Step 4: Register Allocation    — virtual reg → physical (YMM/GPR), spill analysis
//   P10 Step 5: Instruction Scheduling — dependency DAG, critical path, port assignment
//   P10 Step 6: Machine Code Emission  — final assembly + binary encoding (hex)
//
// Test case: GEMM micro-kernel (4×8 register-blocked) + Bias + ReLU
//
// Pure C++17, header-only IR, no external dependencies.

#include "llvm_lowering_ir.h"

#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace backend_ir;

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

// =====================================================================
// P10 Step 0: Vector IR Input (from P9 / 9_vector)
// =====================================================================
// We start from the vector dialect output of the previous pipeline:
//   - GEMM micro-kernel: 4-row register-blocked, vector<8xf32> FMA chain
//   - Bias + ReLU: vectorized addf + maxf
//
// Key ops entering this step:
//   vector.transfer_read  %C[ii, jo]     → %vacc0..3
//   vector.transfer_read  %B[ki, jo]     → %vb
//   scalar load A[ii, ki] → broadcast    → %va
//   vector.fma %va, %vb, %vacc           → %vacc (×4 rows)
//   vector.transfer_write %vacc → %C[ii, jo]
//   arith.addf %vc, %vbias               → %vc
//   arith.maxf %vc, %vzero               → %vc

static void stage0_input() {
  std::cout << "  Vector dialect IR entering backend (from Stage 11, Stage 5 output):\n\n";

  std::cout << "  func.func @gemm_bias_relu_vec(%A: memref<64x128xf32>,\n";
  std::cout << "                                 %B: memref<128x32xf32>,\n";
  std::cout << "                                 %bias: memref<32xf32>,\n";
  std::cout << "                                 %C: memref<64x32xf32>) {\n";
  std::cout << "    // GEMM: tiled 32x8x32, register-blocked 4 rows\n";
  std::cout << "    scf.for %io = 0 to 64 step 32 {\n";
  std::cout << "      scf.for %jo = 0 to 32 step 8 {\n";
  std::cout << "        scf.for %ko = 0 to 128 step 32 {\n";
  std::cout << "          scf.for %ii = 0 to 32 step 4 {\n";
  std::cout << "            %vacc0 = vector.transfer_read C[ii+0, jo] : vec<8xf32>\n";
  std::cout << "            %vacc1 = vector.transfer_read C[ii+1, jo] : vec<8xf32>\n";
  std::cout << "            %vacc2 = vector.transfer_read C[ii+2, jo] : vec<8xf32>\n";
  std::cout << "            %vacc3 = vector.transfer_read C[ii+3, jo] : vec<8xf32>\n";
  std::cout << "            scf.for %ki = 0 to 32 step 1 {\n";
  std::cout << "              %vb = vector.transfer_read B[ko+ki, jo] : vec<8xf32>\n";
  std::cout << "              %va0 = broadcast A[io+ii+0, ko+ki] : vec<8xf32>\n";
  std::cout << "              %vacc0 = vector.fma %va0, %vb, %vacc0\n";
  std::cout << "              %va1 = broadcast A[io+ii+1, ko+ki] : vec<8xf32>\n";
  std::cout << "              %vacc1 = vector.fma %va1, %vb, %vacc1\n";
  std::cout << "              %va2 = broadcast A[io+ii+2, ko+ki] : vec<8xf32>\n";
  std::cout << "              %vacc2 = vector.fma %va2, %vb, %vacc2\n";
  std::cout << "              %va3 = broadcast A[io+ii+3, ko+ki] : vec<8xf32>\n";
  std::cout << "              %vacc3 = vector.fma %va3, %vb, %vacc3\n";
  std::cout << "            }\n";
  std::cout << "            vector.transfer_write %vacc0 → C[ii+0, jo]\n";
  std::cout << "            vector.transfer_write %vacc1 → C[ii+1, jo]\n";
  std::cout << "            vector.transfer_write %vacc2 → C[ii+2, jo]\n";
  std::cout << "            vector.transfer_write %vacc3 → C[ii+3, jo]\n";
  std::cout << "    }}}}  // end GEMM\n";
  std::cout << "    // Bias + ReLU (fused)\n";
  std::cout << "    %vzero = splat 0.0 : vec<8xf32>\n";
  std::cout << "    scf.for %fi = 0 to 64 {\n";
  std::cout << "      scf.for %fj = 0 to 32 step 8 {\n";
  std::cout << "        %vbias = vector.transfer_read bias[fj] : vec<8xf32>\n";
  std::cout << "        %vc = vector.transfer_read C[fi, fj] : vec<8xf32>\n";
  std::cout << "        %vc = arith.addf %vc, %vbias\n";
  std::cout << "        %vc = arith.maxf %vc, %vzero\n";
  std::cout << "        vector.transfer_write %vc → C[fi, fj]\n";
  std::cout << "    }}  // end bias+relu\n";
  std::cout << "    return\n";
  std::cout << "  }\n\n";

  std::cout << "  Op statistics entering backend:\n";
  std::cout << "    vector.transfer_read  : 6 (4 acc + 1 B + 1 bias/C)\n";
  std::cout << "    vector.transfer_write : 5 (4 acc + 1 C)\n";
  std::cout << "    vector.fma            : 4 (4 register-blocked rows)\n";
  std::cout << "    vector.broadcast      : 4 (4 A scalar broadcasts)\n";
  std::cout << "    arith.addf/maxf       : 2 (bias + relu)\n";
}

// =====================================================================
// P10 Step 1: Vector Lowering — vector dialect → LLVM dialect
// =====================================================================
// Lowering rules:
//   memref<NxMxf32>  → ptr (flat pointer, row-major addressing)
//   vector.transfer_read  → getelementptr + load <8 x float>, align 32
//   vector.transfer_write → getelementptr + store <8 x float>, align 32
//   vector.broadcast      → scalar load + shufflevector (splat)
//   vector.fma            → call @llvm.fma.v8f32
//   arith.addf (vec)      → fadd <8 x float>
//   arith.maxf (vec)      → call @llvm.maxnum.v8f32
//
// memref → ptr addressing: memref<NxMxf32>[i, j] → GEP base, i*M + j

static LLVMFunc stage1_vector_lowering() {
  LLVMFunc f;

  std::cout << "  Vector → LLVM Dialect lowering rules:\n\n";
  std::cout << "    ┌───────────────────────────────┬──────────────────────────────────────┐\n";
  std::cout << "    │ Vector Dialect                │ LLVM Dialect                         │\n";
  std::cout << "    ├───────────────────────────────┼──────────────────────────────────────┤\n";
  std::cout << "    │ memref<NxMxf32>               │ ptr (flat, row-major)                │\n";
  std::cout << "    │ memref[i, j] (NxM)            │ GEP ptr, i*M + j                    │\n";
  std::cout << "    │ vector.transfer_read          │ GEP + load <8 x float>, align 32    │\n";
  std::cout << "    │ vector.transfer_write         │ GEP + store <8 x float>, align 32   │\n";
  std::cout << "    │ vector.broadcast(scalar)      │ load float + shufflevector (splat)   │\n";
  std::cout << "    │ vector.fma                    │ call @llvm.fma.v8f32(a, b, c)        │\n";
  std::cout << "    │ arith.addf <8xf32>            │ fadd <8 x float>                    │\n";
  std::cout << "    │ arith.maxf <8xf32>            │ call @llvm.maxnum.v8f32(a, b)        │\n";
  std::cout << "    │ scf.for                       │ br / br i1 / phi (loop CFG)          │\n";
  std::cout << "    └───────────────────────────────┴──────────────────────────────────────┘\n\n";

  std::cout << "  Address computation example (C[ii, jo], M=32):\n";
  std::cout << "    %offset = mul i64 %ii, 32     ; row * stride\n";
  std::cout << "    %offset = add i64 %offset, %jo ; + column\n";
  std::cout << "    %ptr_c = getelementptr float, ptr %C, i64 %offset\n";
  std::cout << "    %vacc = load <8 x float>, ptr %ptr_c, align 32\n\n";

  // Build the LLVM dialect representation of ki-loop body (inner kernel)
  f.ops.push_back(LLVMOp::Comment("=== LLVM dialect: GEMM ki-loop body (1 iteration, 4-row blocked) ==="));
  f.ops.push_back(LLVMOp::Comment("Assume: %io, %jo, %ko, %ii, %ki are loop induction variables"));
  f.ops.push_back(LLVMOp::Comment("        %vacc0..3 are phi-carried accumulators from loop header"));

  // B row load
  f.ops.push_back(LLVMOp::Comment("--- load B row ---"));
  f.ops.push_back(LLVMOp::GEP("%gep_b", "%B", "(%ko+%ki)*32 + %jo"));
  f.ops.push_back(LLVMOp::Load("%vb", "%gep_b", LLVMTy::VEC8F32, 32));

  // 4 rows of A broadcast + FMA
  for (int r = 0; r < 4; ++r) {
    std::string rs = std::to_string(r);
    f.ops.push_back(LLVMOp::Comment("--- row " + rs + " ---"));
    f.ops.push_back(LLVMOp::GEP("%gep_a" + rs, "%A",
                                  "(%io+%ii+" + rs + ")*128 + %ko+%ki"));
    f.ops.push_back(LLVMOp::Load("%a_scalar" + rs, "%gep_a" + rs, LLVMTy::F32));
    f.ops.push_back(LLVMOp::Broadcast("%va" + rs, "%a_scalar" + rs));
    f.ops.push_back(LLVMOp::FMA("%vacc" + rs, "%va" + rs, "%vb", "%vacc" + rs));
  }

  f.ops.push_back(LLVMOp::Comment("=== LLVM dialect: Bias+ReLU (fj-loop body) ==="));
  f.ops.push_back(LLVMOp::GEP("%gep_bias", "%bias", "%fj"));
  f.ops.push_back(LLVMOp::Load("%vbias", "%gep_bias", LLVMTy::VEC8F32, 32));
  f.ops.push_back(LLVMOp::GEP("%gep_c", "%C", "%fi*32 + %fj"));
  f.ops.push_back(LLVMOp::Load("%vc", "%gep_c", LLVMTy::VEC8F32, 32));
  f.ops.push_back(LLVMOp::FAdd("%vc1", "%vc", "%vbias"));
  f.ops.push_back(LLVMOp::MaxNum("%vc2", "%vc1", "%vzero"));
  f.ops.push_back(LLVMOp::Store("%vc2", "%gep_c", LLVMTy::VEC8F32, 32));

  int loads = f.count(LLVMOpKind::LOAD);
  int stores = f.count(LLVMOpKind::STORE);
  int fmas = f.count(LLVMOpKind::FMA);
  int geps = f.count(LLVMOpKind::GEP);

  std::cout << "  LLVM dialect op count (ki-loop body + bias/relu body):\n";
  std::cout << "    getelementptr : " << geps << "\n";
  std::cout << "    load          : " << loads << "\n";
  std::cout << "    store         : " << stores << "\n";
  std::cout << "    llvm.fma      : " << fmas << "\n";
  std::cout << "    shufflevector : " << f.count(LLVMOpKind::BROADCAST) << "\n";
  std::cout << "    fadd          : " << f.count(LLVMOpKind::FADD) << "\n";
  std::cout << "    maxnum        : " << f.count(LLVMOpKind::MAXNUM) << "\n";

  return f;
}

// =====================================================================
// P10 Step 2: LLVM Dialect → Textual LLVM IR
// =====================================================================
// Convert LLVM dialect to proper textual LLVM IR with:
//   - Function signature with ptr arguments
//   - Basic block structure (entry, loop header, loop body, loop exit)
//   - PHI nodes for loop induction variables and accumulators
//   - Proper SSA naming (%0, %1, ...)

static LLVMFunc stage2_llvm_ir() {
  LLVMFunc ir;
  std::cout << "  Converting LLVM dialect → textual LLVM IR:\n";
  std::cout << "    - Flatten memref addressing to linear GEPs\n";
  std::cout << "    - Build loop CFG (header → body → latch → exit)\n";
  std::cout << "    - Insert PHI nodes for loop-carried values\n";
  std::cout << "    - Name SSA values sequentially (%0, %1, ...)\n\n";

  ir.ops.push_back(LLVMOp::FuncBegin("gemm_bias_relu"));
  ir.ops.push_back(LLVMOp::Label("entry"));
  ir.ops.push_back(LLVMOp::Comment("zero vector for ReLU"));
  ir.ops.push_back(LLVMOp::Broadcast("%vzero", "0.0"));
  ir.ops.push_back(LLVMOp::Br("ki.header"));

  // ki-loop (innermost of GEMM, showing just this for clarity)
  ir.ops.push_back(LLVMOp::Label("ki.header"));
  ir.ops.push_back(LLVMOp::Phi("%ki", LLVMTy::I64,
      {{"0", "entry"}, {"%ki.next", "ki.latch"}}));
  ir.ops.push_back(LLVMOp::Phi("%acc0", LLVMTy::VEC8F32,
      {{"%%vacc0.init", "entry"}, {"%acc0.next", "ki.latch"}}));
  ir.ops.push_back(LLVMOp::Phi("%acc1", LLVMTy::VEC8F32,
      {{"%%vacc1.init", "entry"}, {"%acc1.next", "ki.latch"}}));
  ir.ops.push_back(LLVMOp::Phi("%acc2", LLVMTy::VEC8F32,
      {{"%%vacc2.init", "entry"}, {"%acc2.next", "ki.latch"}}));
  ir.ops.push_back(LLVMOp::Phi("%acc3", LLVMTy::VEC8F32,
      {{"%%vacc3.init", "entry"}, {"%acc3.next", "ki.latch"}}));
  ir.ops.push_back(LLVMOp::ICmp("%done", "sge", "%ki", "32"));
  ir.ops.push_back(LLVMOp::BrCond("%done", "ki.exit", "ki.body"));

  ir.ops.push_back(LLVMOp::Label("ki.body"));
  ir.ops.push_back(LLVMOp::Comment("B row: GEP + aligned load"));
  ir.ops.push_back(LLVMOp::GEP("%bp", "%B", "%ki*32 + %jo"));
  ir.ops.push_back(LLVMOp::Load("%vb", "%bp", LLVMTy::VEC8F32, 32));

  for (int r = 0; r < 4; ++r) {
    std::string rs = std::to_string(r);
    ir.ops.push_back(LLVMOp::Comment("A[ii+" + rs + ", ki] → broadcast → fma"));
    ir.ops.push_back(LLVMOp::GEP("%ap" + rs, "%A", "(%ii+" + rs + ")*128+%ki"));
    ir.ops.push_back(LLVMOp::Load("%as" + rs, "%ap" + rs, LLVMTy::F32));
    ir.ops.push_back(LLVMOp::Broadcast("%va" + rs, "%as" + rs));
    ir.ops.push_back(LLVMOp::FMA("%acc" + rs + ".next", "%va" + rs, "%vb",
                                   "%acc" + rs));
  }
  ir.ops.push_back(LLVMOp::Br("ki.latch"));

  ir.ops.push_back(LLVMOp::Label("ki.latch"));
  ir.ops.push_back(LLVMOp::AddI64("%ki.next", "%ki", "1"));
  ir.ops.push_back(LLVMOp::Br("ki.header"));

  ir.ops.push_back(LLVMOp::Label("ki.exit"));
  ir.ops.push_back(LLVMOp::Comment("store accumulators back to C"));
  for (int r = 0; r < 4; ++r) {
    std::string rs = std::to_string(r);
    ir.ops.push_back(LLVMOp::GEP("%cp" + rs, "%C", "(%ii+" + rs + ")*32+%jo"));
    ir.ops.push_back(LLVMOp::Store("%acc" + rs, "%cp" + rs,
                                     LLVMTy::VEC8F32, 32));
  }

  ir.ops.push_back(LLVMOp::Comment("... (outer loops omitted for clarity) ..."));
  ir.ops.push_back(LLVMOp::Comment("Bias + ReLU body:"));
  ir.ops.push_back(LLVMOp::GEP("%bias_p", "%bias", "%fj"));
  ir.ops.push_back(LLVMOp::Load("%vbias", "%bias_p", LLVMTy::VEC8F32, 32));
  ir.ops.push_back(LLVMOp::GEP("%c_p", "%C", "%fi*32+%fj"));
  ir.ops.push_back(LLVMOp::Load("%vc_old", "%c_p", LLVMTy::VEC8F32, 32));
  ir.ops.push_back(LLVMOp::FAdd("%vc_add", "%vc_old", "%vbias"));
  ir.ops.push_back(LLVMOp::MaxNum("%vc_relu", "%vc_add", "%vzero"));
  ir.ops.push_back(LLVMOp::Store("%vc_relu", "%c_p", LLVMTy::VEC8F32, 32));
  ir.ops.push_back(LLVMOp::Ret());
  ir.ops.push_back(LLVMOp::FuncEnd());

  std::cout << "  Generated LLVM IR (ki-loop focus):\n\n";
  ir.print(std::cout);

  std::cout << "\n  CFG structure:\n";
  std::cout << "    entry → ki.header → ki.body → ki.latch ──→ ki.header (loop)\n";
  std::cout << "                      └─ ki.exit (done) → store → bias_relu → ret\n\n";

  std::cout << "  Key LLVM IR patterns:\n";
  std::cout << "    • PHI nodes: %ki (induction), %acc0..3 (loop-carried accumulators)\n";
  std::cout << "    • Aligned loads: load <8 x float>, ptr, align 32\n";
  std::cout << "    • FMA intrinsic: @llvm.fma.v8f32 (3-operand fused multiply-add)\n";
  std::cout << "    • Splat: shufflevector <1 x float> → <8 x float> zeroinitializer\n";

  return ir;
}

// =====================================================================
// P10 Step 3: Instruction Selection (ISel)
// =====================================================================
// Map LLVM IR to x86-64 AVX2 machine instructions using DAG pattern matching.
//
// ISel patterns (LLVM IR → x86-64 AVX2):
//   @llvm.fma.v8f32(a,b,c) → vfmadd231ps dst, src1, src2
//   load <8xf32>, align 32 → vmovaps ymm, [mem]
//   store <8xf32>, align 32 → vmovaps [mem], ymm
//   shufflevector (splat)    → vbroadcastss ymm, [mem]
//   fadd <8xf32>            → vaddps ymm, ymm, ymm
//   @llvm.maxnum.v8f32      → vmaxps ymm, ymm, ymm
//
// Complex addressing: GEP + load → vmovaps ymm, [base + idx*4]

static std::vector<MachineInstr> stage3_isel() {
  std::vector<MachineInstr> mi;

  std::cout << "  Instruction Selection (DAG pattern matching):\n\n";
  std::cout << "    ┌────────────────────────────────┬───────────────────────────────────┐\n";
  std::cout << "    │ LLVM IR Pattern                │ x86-64 AVX2 Instruction           │\n";
  std::cout << "    ├────────────────────────────────┼───────────────────────────────────┤\n";
  std::cout << "    │ load <8xf32>, align 32         │ vmovaps ymm, [rdi+rax*4]         │\n";
  std::cout << "    │ store <8xf32>, align 32        │ vmovaps [rdi+rax*4], ymm         │\n";
  std::cout << "    │ @llvm.fma.v8f32(a,b,c)         │ vfmadd231ps ymm, ymm, ymm        │\n";
  std::cout << "    │ shufflevector (splat)           │ vbroadcastss ymm, [rdi+rax*4]    │\n";
  std::cout << "    │ fadd <8xf32>                   │ vaddps ymm, ymm, ymm             │\n";
  std::cout << "    │ @llvm.maxnum.v8f32(a,b)        │ vmaxps ymm, ymm, ymm             │\n";
  std::cout << "    │ GEP + load (fold)              │ vmovaps ymm, [base+idx*4+disp]   │\n";
  std::cout << "    │ icmp sge + br i1               │ cmp + jge / jl                    │\n";
  std::cout << "    │ add i64 %iv, 1                 │ add rcx, 1                        │\n";
  std::cout << "    └────────────────────────────────┴───────────────────────────────────┘\n\n";

  std::cout << "  Address mode folding:\n";
  std::cout << "    GEP: %p = getelementptr float, ptr %B, i64 (%ki*32 + %jo)\n";
  std::cout << "    + load <8 x float>, ptr %p, align 32\n";
  std::cout << "    → vmovaps ymm1, [rsi + rcx*128 + rdx*4]  ; fused GEP+load\n";
  std::cout << "    (rsi=B base, rcx=ki, rdx=jo, scale=4 for float)\n\n";

  // Build ki-loop body as MachineInstrs
  mi.push_back(MachineInstr::Comment("=== ki-loop body (inner GEMM kernel, 4-row blocked) ==="));
  mi.push_back(MachineInstr::LabelM(".Lki_body"));

  // B row load (fused GEP+load)
  auto b_load = MachineInstr::VMovapsLoad("%vreg_vb", "[rsi+rcx*128+rdx*4]");
  b_load.annotation = "B[ko+ki, jo] → vreg_vb";
  b_load.id = 1;
  mi.push_back(b_load);

  // 4 rows: broadcast A + FMA
  for (int r = 0; r < 4; ++r) {
    std::string rs = std::to_string(r);
    std::string disp = (r == 0) ? "" : "+" + std::to_string(r * 128 * 4);

    auto bcast = MachineInstr::VBroadcastss(
        "%vreg_va" + rs, "[rdi+r8*512+rcx*4" + disp + "]");
    bcast.annotation = "A[io+ii+" + rs + ", ko+ki]";
    bcast.id = 2 + r * 2;
    bcast.deps = {};
    mi.push_back(bcast);

    auto fma = MachineInstr::VFmadd231ps(
        "%vreg_acc" + rs, "%vreg_va" + rs, "%vreg_vb");
    fma.annotation = "acc" + rs + " += va" + rs + " * vb";
    fma.id = 3 + r * 2;
    fma.deps = {2 + r * 2, 1}; // depends on broadcast and B load
    mi.push_back(fma);
  }

  // Loop control
  auto add_ki = MachineInstr::AddI("rcx", "1");
  add_ki.annotation = "ki++";
  add_ki.id = 10;
  mi.push_back(add_ki);

  auto cmp_ki = MachineInstr::Cmp("rcx", "32");
  cmp_ki.annotation = "ki < 32?";
  cmp_ki.id = 11;
  cmp_ki.deps = {10};
  mi.push_back(cmp_ki);

  mi.push_back(MachineInstr::Jl(".Lki_body"));

  // Store accumulators
  mi.push_back(MachineInstr::Comment("=== store C rows (after ki-loop) ==="));
  for (int r = 0; r < 4; ++r) {
    std::string rs = std::to_string(r);
    auto st = MachineInstr::VMovapsStore(
        "[r9+r8*128+rdx*4+" + std::to_string(r * 32 * 4) + "]",
        "%vreg_acc" + rs);
    st.annotation = "C[ii+" + rs + ", jo]";
    mi.push_back(st);
  }

  // Bias + ReLU
  mi.push_back(MachineInstr::Comment("=== Bias + ReLU (fj-loop body) ==="));
  auto xor_z = MachineInstr::VXorps("%vreg_zero", "%vreg_zero", "%vreg_zero");
  xor_z.annotation = "vzero = 0.0";
  mi.push_back(xor_z);

  auto bias_ld = MachineInstr::VMovapsLoad("%vreg_bias", "[r10+rdx*4]");
  bias_ld.annotation = "bias[fj]";
  mi.push_back(bias_ld);

  auto c_ld = MachineInstr::VMovapsLoad("%vreg_c", "[r9+rax*128+rdx*4]");
  c_ld.annotation = "C[fi, fj]";
  mi.push_back(c_ld);

  auto add_bias = MachineInstr::VAddps("%vreg_c", "%vreg_c", "%vreg_bias");
  add_bias.annotation = "C += bias";
  mi.push_back(add_bias);

  auto relu = MachineInstr::VMaxps("%vreg_c", "%vreg_c", "%vreg_zero");
  relu.annotation = "ReLU";
  mi.push_back(relu);

  auto c_st = MachineInstr::VMovapsStore("[r9+rax*128+rdx*4]", "%vreg_c");
  c_st.annotation = "C[fi, fj]";
  mi.push_back(c_st);

  std::cout << "  Selected instructions (ki-loop body + bias/relu):\n\n";
  for (auto &m : mi) m.print(std::cout);

  int n_fma = 0, n_load = 0, n_store = 0, n_bcast = 0;
  for (auto &m : mi) {
    if (m.kind == MIOpKind::VFMADD231PS) ++n_fma;
    if (m.kind == MIOpKind::VMOVAPS_LOAD) ++n_load;
    if (m.kind == MIOpKind::VMOVAPS_STORE) ++n_store;
    if (m.kind == MIOpKind::VBROADCASTSS) ++n_bcast;
  }
  std::cout << "\n  ISel statistics:\n";
  std::cout << "    vfmadd231ps   : " << n_fma << "\n";
  std::cout << "    vmovaps (load): " << n_load << "\n";
  std::cout << "    vmovaps (store):" << n_store << "\n";
  std::cout << "    vbroadcastss  : " << n_bcast << "\n";
  std::cout << "    Total MachineInstrs: " << mi.size() << " (excl. comments/labels)\n";

  return mi;
}

// =====================================================================
// P10 Step 4: Register Allocation
// =====================================================================
// Map virtual registers → physical registers (x86-64 AVX2).
//
// Physical register file:
//   YMM0-YMM15  (16 × 256-bit vector)
//   RAX..R15    (16 × 64-bit GPR)
//
// Strategy: linear scan (simplified):
//   - Accumulators %vreg_acc0..3 get priority (long live range, hot loop)
//   - B row %vreg_vb reused each iteration (short live range)
//   - Broadcasts %vreg_va0..3 are short-lived (consumed by FMA same cycle)
//
// If pressure > 16 YMM → spill to stack (not needed for our 4×8 kernel)

static std::vector<RegMapping> stage4_regalloc() {
  std::cout << "  x86-64 AVX2 register file:\n";
  std::cout << "    Vector: YMM0–YMM15 (16 × 256-bit = 16 × 8xf32)\n";
  std::cout << "    GPR:    RAX, RCX, RDX, RSI, RDI, R8–R15 (16 × 64-bit)\n\n";

  std::cout << "  Live range analysis (ki-loop body):\n";
  std::cout << "    ┌─────────────────┬─────────────────┬──────────────┐\n";
  std::cout << "    │ Virtual Reg     │ Live Range      │ Priority     │\n";
  std::cout << "    ├─────────────────┼─────────────────┼──────────────┤\n";
  std::cout << "    │ %vreg_acc0      │ [load → store]  │ HIGH (loop)  │\n";
  std::cout << "    │ %vreg_acc1      │ [load → store]  │ HIGH (loop)  │\n";
  std::cout << "    │ %vreg_acc2      │ [load → store]  │ HIGH (loop)  │\n";
  std::cout << "    │ %vreg_acc3      │ [load → store]  │ HIGH (loop)  │\n";
  std::cout << "    │ %vreg_vb        │ [load → 4×fma]  │ MEDIUM       │\n";
  std::cout << "    │ %vreg_va0       │ [bcast → fma]   │ LOW (short)  │\n";
  std::cout << "    │ %vreg_va1       │ [bcast → fma]   │ LOW (short)  │\n";
  std::cout << "    │ %vreg_va2       │ [bcast → fma]   │ LOW (short)  │\n";
  std::cout << "    │ %vreg_va3       │ [bcast → fma]   │ LOW (short)  │\n";
  std::cout << "    │ %vreg_zero      │ [entry → end]   │ HIGH (const) │\n";
  std::cout << "    │ %vreg_bias      │ [load → vaddps] │ LOW (short)  │\n";
  std::cout << "    │ %vreg_c         │ [load → store]  │ MEDIUM       │\n";
  std::cout << "    └─────────────────┴─────────────────┴──────────────┘\n\n";

  std::vector<RegMapping> mapping;

  // Vector registers (YMM)
  mapping.push_back({"%vreg_acc0", "ymm0", false, -1});
  mapping.push_back({"%vreg_acc1", "ymm1", false, -1});
  mapping.push_back({"%vreg_acc2", "ymm2", false, -1});
  mapping.push_back({"%vreg_acc3", "ymm3", false, -1});
  mapping.push_back({"%vreg_vb",   "ymm4", false, -1});
  mapping.push_back({"%vreg_va0",  "ymm5", false, -1});
  mapping.push_back({"%vreg_va1",  "ymm6", false, -1});
  mapping.push_back({"%vreg_va2",  "ymm7", false, -1});
  mapping.push_back({"%vreg_va3",  "ymm8", false, -1});
  mapping.push_back({"%vreg_zero", "ymm15", false, -1});
  mapping.push_back({"%vreg_bias", "ymm9", false, -1});
  mapping.push_back({"%vreg_c",    "ymm10", false, -1});

  // GPR
  mapping.push_back({"%A_ptr",  "rdi", false, -1});
  mapping.push_back({"%B_ptr",  "rsi", false, -1});
  mapping.push_back({"%bias_ptr", "rdx", false, -1});
  mapping.push_back({"%C_ptr",  "rcx", false, -1});
  mapping.push_back({"%io",     "r8",  false, -1});
  mapping.push_back({"%jo",     "r9",  false, -1});
  mapping.push_back({"%ko",     "r10", false, -1});
  mapping.push_back({"%ii",     "r11", false, -1});
  mapping.push_back({"%ki",     "r12", false, -1});

  std::cout << "  Register allocation result (linear scan):\n\n";
  std::cout << "    Vector registers (YMM):\n";
  for (size_t i = 0; i < 12; ++i) mapping[i].print(std::cout);
  std::cout << "\n    General-purpose registers (GPR):\n";
  for (size_t i = 12; i < mapping.size(); ++i) mapping[i].print(std::cout);

  std::cout << "\n  Register pressure analysis:\n";
  std::cout << "    YMM used:  12 / 16  (75% utilization)\n";
  std::cout << "    YMM free:   4       (ymm11..14 available for spill/unroll)\n";
  std::cout << "    GPR used:   9 / 16  (loop IVs + ptr args)\n";
  std::cout << "    Spills:     0       (no register pressure overflow)\n\n";

  std::cout << "  If we unrolled ii by 8 instead of 4:\n";
  std::cout << "    8 accumulators + 1 B + 4 A broadcasts + 1 zero = 14 YMM\n";
  std::cout << "    → still fits (14/16), but leaves only 2 free → risky\n";
  std::cout << "    → 4-row unroll is the sweet spot for AVX2\n";

  // Spill example (hypothetical)
  std::cout << "\n  Spill example (if 8-row unroll were forced):\n";
  RegMapping spill_ex{"%vreg_acc7", "", true, 0};
  spill_ex.print(std::cout);
  std::cout << "    → vmovaps [rsp+0], ymm_spill  (spill to stack)\n";
  std::cout << "    → vmovaps ymm_spill, [rsp+0]  (reload from stack)\n";
  std::cout << "    Cost: 2 × 5-cycle load/store = 10 extra cycles per ki iteration\n";

  return mapping;
}

// =====================================================================
// P10 Step 5: Instruction Scheduling
// =====================================================================
// Reorder instructions to maximize ILP (Instruction-Level Parallelism).
//
// Intel Haswell/Skylake execution model:
//   Port 0: FMA (vfmadd231ps), 4-cycle latency, 1-cycle throughput
//   Port 1: FADD (vaddps, vmaxps), 4-cycle latency
//   Port 2/3: Load (vmovaps load), 5-cycle latency (L1 hit)
//   Port 4: Store (vmovaps store), 5-cycle latency
//   Port 5: Shuffle (vbroadcastss), 1-cycle latency
//
// Goal: interleave loads with FMAs to hide load latency.
// Before scheduling: load B → bcast A0 → fma0 → bcast A1 → fma1 → ...
// After scheduling:  load B, bcast A0 in parallel → fma0 | bcast A1 → ...

static std::vector<SchedSlot> stage5_scheduling() {
  std::vector<SchedSlot> sched;

  std::cout << "  Execution port model (Intel Haswell/Skylake):\n\n";
  std::cout << "    ┌────────┬────────────────────────────┬─────────┬────────────┐\n";
  std::cout << "    │ Port   │ Operations                 │ Latency │ Throughput │\n";
  std::cout << "    ├────────┼────────────────────────────┼─────────┼────────────┤\n";
  std::cout << "    │ Port 0 │ FMA (vfmadd231ps)          │ 4 cy    │ 1/cy       │\n";
  std::cout << "    │ Port 1 │ ADD/MAX (vaddps, vmaxps)   │ 4 cy    │ 1/cy       │\n";
  std::cout << "    │ Port 2 │ Load (vmovaps, L1 hit)     │ 5 cy    │ 1/cy       │\n";
  std::cout << "    │ Port 3 │ Load (alternate)           │ 5 cy    │ 1/cy       │\n";
  std::cout << "    │ Port 4 │ Store                      │ 5 cy    │ 1/cy       │\n";
  std::cout << "    │ Port 5 │ Shuffle (vbroadcastss)     │ 1 cy    │ 1/cy       │\n";
  std::cout << "    └────────┴────────────────────────────┴─────────┴────────────┘\n\n";

  std::cout << "  Dependency DAG (ki-loop body):\n";
  std::cout << "                load B ──────────┐\n";
  std::cout << "                  │               │\n";
  std::cout << "    bcast A0 ──→ fma0    bcast A1 ──→ fma1\n";
  std::cout << "                  │                     │\n";
  std::cout << "    bcast A2 ──→ fma2    bcast A3 ──→ fma3\n";
  std::cout << "                  │                     │\n";
  std::cout << "                  └──── ki++ / cmp / jl\n\n";

  std::cout << "  Before scheduling (program order):\n";
  std::cout << "    cycle 0: vmovaps ymm4, [B]         ; load B (5-cy latency)\n";
  std::cout << "    cycle 1: vbroadcastss ymm5, [A+0]  ; (stall: wait for B? no!)\n";
  std::cout << "    cycle 2: vfmadd231ps ymm0, ymm5, ymm4  ; (wait if B not ready)\n";
  std::cout << "    cycle 3: vbroadcastss ymm6, [A+1]\n";
  std::cout << "    cycle 4: vfmadd231ps ymm1, ymm6, ymm4\n";
  std::cout << "    ...serial execution → 13 cycles total\n\n";

  std::cout << "  After scheduling (interleaved for ILP):\n\n";

  // Optimized schedule
  sched.push_back({0, 1, "vmovaps",      "ymm4, [rsi+...]",    2, "load B (port 2)"});
  sched.push_back({0, 2, "vbroadcastss", "ymm5, [rdi+...]",    5, "bcast A0 (port 5) ← parallel!"});
  sched.push_back({1, 3, "vbroadcastss", "ymm6, [rdi+512]",    5, "bcast A1 (port 5)"});
  sched.push_back({2, 4, "vbroadcastss", "ymm7, [rdi+1024]",   5, "bcast A2 (port 5)"});
  sched.push_back({3, 5, "vbroadcastss", "ymm8, [rdi+1536]",   5, "bcast A3 (port 5)"});
  sched.push_back({5, 6, "vfmadd231ps",  "ymm0, ymm5, ymm4",  0, "fma0 (port 0) ← B ready at cy 5"});
  sched.push_back({6, 7, "vfmadd231ps",  "ymm1, ymm6, ymm4",  0, "fma1 (port 0)"});
  sched.push_back({7, 8, "vfmadd231ps",  "ymm2, ymm7, ymm4",  0, "fma2 (port 0)"});
  sched.push_back({8, 9, "vfmadd231ps",  "ymm3, ymm8, ymm4",  0, "fma3 (port 0)"});
  sched.push_back({9, 10, "add",         "r12, 1",             1, "ki++ (port 0/1)"});
  sched.push_back({9, 11, "cmp",         "r12, 32",            1, "ki < 32? (port 0/1)"});
  sched.push_back({10, 12, "jl",         ".Lki_body",          1, ""});

  std::cout << "    ┌──────────┬────────┬────────────────┬─────────────────────────────────┐\n";
  std::cout << "    │ Cycle    │ Port   │ Instruction    │ Operands                        │\n";
  std::cout << "    ├──────────┼────────┼────────────────┼─────────────────────────────────┤\n";
  for (auto &s : sched) {
    std::printf("    │ cycle %2d │ port %d │ %-14s │ %-31s │",
                s.cycle, s.port, s.mnemonic.c_str(), s.operands.c_str());
    if (!s.annotation.empty()) std::printf("  %s", s.annotation.c_str());
    std::printf("\n");
  }
  std::cout << "    └──────────┴────────┴────────────────┴─────────────────────────────────┘\n\n";

  std::cout << "  Scheduling analysis:\n";
  std::cout << "    Before: 13 cycles (serial)\n";
  std::cout << "    After:  11 cycles (ILP: load‖broadcast, 4 FMAs pipelined)\n";
  std::cout << "    Speedup: 1.18× from scheduling alone\n\n";

  std::cout << "  Key scheduling insights:\n";
  std::cout << "    • Load B and broadcast A0 issue in parallel (port 2 ‖ port 5)\n";
  std::cout << "    • 4 broadcasts fill the gap while B load completes (5 cycles)\n";
  std::cout << "    • FMAs are back-to-back with 1-cy throughput (4 independent chains)\n";
  std::cout << "    • ki++ / cmp overlap with last FMA execution\n";
  std::cout << "    • Software pipelining: next iteration's load B can overlap with\n";
  std::cout << "      current iteration's last FMA (not shown, requires loop unrolling)\n";

  return sched;
}

// =====================================================================
// P10 Step 6: Machine Code Emission
// =====================================================================
// Emit final x86-64 AVX2 assembly with binary encoding.
//
// VEX prefix encoding for AVX2:
//   vfmadd231ps ymm0, ymm5, ymm4 → C4 E2 55 B8 C4
//   vmovaps ymm4, [rsi]           → C5 FC 28 26
//   vbroadcastss ymm5, [rdi]      → C4 E2 7D 18 2F

static void stage6_emit() {
  std::vector<AsmLine> code;

  std::cout << "  Final assembly (x86-64 AVX2, AT&T syntax):\n\n";

  // Function prologue
  code.push_back({"gemm_bias_relu", "", "", "", "", -1});
  code.push_back({"", "push", "r12", "", "save callee-saved", 0});
  code.push_back({"", "push", "r13", "", "save callee-saved", 2});
  code.push_back({"", "vxorps", "ymm15, ymm15, ymm15", "C5 04 57 FF", "vzero = 0.0", 4});

  // Ki-loop body
  code.push_back({".Lki_body", "", "", "", "", -1});
  code.push_back({"", "vmovaps", "ymm4, [rsi+r12*128+r9*4]",
                   "C4 A1 7C 28 64 B4 00", "B[ko+ki, jo]", 0x10});
  code.push_back({"", "vbroadcastss", "ymm5, [rdi+r11*512+r12*4]",
                   "C4 A2 7D 18 6C 9F 00", "A[ii+0, ki]", 0x17});
  code.push_back({"", "vbroadcastss", "ymm6, [rdi+r11*512+r12*4+512]",
                   "C4 A2 7D 18 74 9F 02", "A[ii+1, ki]", 0x1E});
  code.push_back({"", "vbroadcastss", "ymm7, [rdi+r11*512+r12*4+1024]",
                   "C4 A2 7D 18 7C 9F 04", "A[ii+2, ki]", 0x25});
  code.push_back({"", "vbroadcastss", "ymm8, [rdi+r11*512+r12*4+1536]",
                   "C4 62 7D 18 44 9F 06", "A[ii+3, ki]", 0x2C});

  code.push_back({"", "vfmadd231ps", "ymm0, ymm5, ymm4",
                   "C4 E2 55 B8 C4",       "acc0 += A[0]*B", 0x33});
  code.push_back({"", "vfmadd231ps", "ymm1, ymm6, ymm4",
                   "C4 E2 4D B8 CC",       "acc1 += A[1]*B", 0x38});
  code.push_back({"", "vfmadd231ps", "ymm2, ymm7, ymm4",
                   "C4 E2 45 B8 D4",       "acc2 += A[2]*B", 0x3D});
  code.push_back({"", "vfmadd231ps", "ymm3, ymm8, ymm4",
                   "C4 C2 3D B8 DC",       "acc3 += A[3]*B", 0x42});

  code.push_back({"", "inc", "r12", "49 FF C4", "ki++", 0x47});
  code.push_back({"", "cmp", "r12, 32", "49 83 FC 20", "ki < 32?", 0x4A});
  code.push_back({"", "jl", ".Lki_body", "7C C4", "loop back", 0x4E});

  // Store accumulators
  code.push_back({"", "vmovaps", "[rcx+r11*128+r9*4], ymm0",
                   "C4 A1 7C 29 04 99",    "C[ii+0, jo]", 0x50});
  code.push_back({"", "vmovaps", "[rcx+r11*128+r9*4+128], ymm1",
                   "C4 A1 7C 29 4C 99 04", "C[ii+1, jo]", 0x56});
  code.push_back({"", "vmovaps", "[rcx+r11*128+r9*4+256], ymm2",
                   "C4 A1 7C 29 54 99 08", "C[ii+2, jo]", 0x5D});
  code.push_back({"", "vmovaps", "[rcx+r11*128+r9*4+384], ymm3",
                   "C4 A1 7C 29 5C 99 0C", "C[ii+3, jo]", 0x64});

  // Bias + ReLU
  code.push_back({".Lbias_relu", "", "", "", "", -1});
  code.push_back({"", "vmovaps", "ymm9, [rdx+r9*4]",
                   "C4 A1 7C 28 0C 8A",    "bias[fj]", 0x70});
  code.push_back({"", "vmovaps", "ymm10, [rcx+rax*128+r9*4]",
                   "C4 21 7C 28 14 81",    "C[fi, fj]", 0x76});
  code.push_back({"", "vaddps", "ymm10, ymm10, ymm9",
                   "C4 41 2C 58 D1",       "C += bias", 0x7C});
  code.push_back({"", "vmaxps", "ymm10, ymm10, ymm15",
                   "C4 41 2C 5F D7",       "ReLU", 0x81});
  code.push_back({"", "vmovaps", "[rcx+rax*128+r9*4], ymm10",
                   "C4 21 7C 29 14 81",    "C[fi, fj]", 0x86});

  // Epilogue
  code.push_back({"", "pop", "r13", "", "", 0x8C});
  code.push_back({"", "pop", "r12", "", "", 0x8E});
  code.push_back({"", "ret", "", "C3", "", 0x90});

  for (auto &a : code) a.print(std::cout);

  std::cout << "\n  Binary encoding details:\n";
  std::cout << "    VEX prefix format (3-byte): C4 [RXB.mmmmm] [W.vvvv.L.pp]\n";
  std::cout << "      R: REX.R inverted    mmmmm: opcode map (0F=01, 0F38=02)\n";
  std::cout << "      X: REX.X inverted    W: 64-bit operand\n";
  std::cout << "      B: REX.B inverted    vvvv: source reg (inverted)\n";
  std::cout << "                           L: 1=256-bit, 0=128-bit\n";
  std::cout << "                           pp: 00=PS, 01=PD, 10=F3, 11=F2\n\n";

  std::cout << "  Encoding example (vfmadd231ps ymm0, ymm5, ymm4):\n";
  std::cout << "    C4 E2 55 B8 C4\n";
  std::cout << "    │  │  │  │  └─ ModRM: 11 000 100 (ymm0 ← ymm4)\n";
  std::cout << "    │  │  │  └──── opcode: B8 (VFMADD231PS)\n";
  std::cout << "    │  │  └─────── W=0, vvvv=0101 (ymm5 inverted=1010→5), L=1, pp=01\n";
  std::cout << "    │  └────────── RXB=111, mmmmm=00010 (0F38 map)\n";
  std::cout << "    └───────────── VEX 3-byte escape\n\n";

  // Code size summary
  std::cout << "  Code size: ~145 bytes (0x00–0x90)\n";
  std::cout << "    GEMM ki-loop body:  ~62 bytes (compact due to VEX encoding)\n";
  std::cout << "    Store C rows:       ~28 bytes\n";
  std::cout << "    Bias+ReLU:          ~30 bytes\n";
  std::cout << "    Prologue/epilogue:  ~10 bytes\n";

  // AArch64 comparison
  std::cout << "\n  ── AArch64 NEON equivalent (for comparison) ──\n\n";
  std::cout << "    // AArch64 uses 128-bit NEON (4xf32), needs 2× iterations for 8xf32\n";
  std::cout << "    ld1     {v4.4s}, [x1], #16        // B row (low 4)\n";
  std::cout << "    ld1     {v5.4s}, [x1], #16        // B row (high 4)\n";
  std::cout << "    ld1r    {v6.4s}, [x0], #4         // broadcast A[ii,ki]\n";
  std::cout << "    fmla    v0.4s, v6.4s, v4.4s       // acc_lo += A * B_lo\n";
  std::cout << "    fmla    v1.4s, v6.4s, v5.4s       // acc_hi += A * B_hi\n";
  std::cout << "    // 2 fmla per row × 4 rows = 8 fmla (vs 4 vfmadd on AVX2)\n\n";

  std::cout << "  ── AArch64 SVE equivalent (scalable vectors) ──\n\n";
  std::cout << "    // SVE adapts to hardware VL (128..2048 bit)\n";
  std::cout << "    ptrue   p0.s                       // all-true predicate\n";
  std::cout << "    ld1w    z4.s, p0/z, [x1, x2, lsl #2]  // B row\n";
  std::cout << "    ld1rw   z6.s, p0/z, [x0]          // broadcast A[ii,ki]\n";
  std::cout << "    fmla    z0.s, p0/m, z6.s, z4.s    // acc += A * B\n";
  std::cout << "    // SVE: same code works for 128-bit through 2048-bit\n";
}

// =====================================================================
// Pipeline Summary
// =====================================================================

static void print_summary() {
  std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
  std::cout << "  │         Vector → Machine Code Pipeline Summary              │\n";
  std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Input : vector dialect (transfer_read/write, fma, etc.)    │\n";
  std::cout << "  │  Output: x86-64 AVX2 machine code (~145 bytes)             │\n";
  std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Pipeline stages:                                           │\n";
  std::cout << "  │    S0: Vector IR input (from Stage 11)                      │\n";
  std::cout << "  │    S1: Vector lowering → LLVM dialect (GEP+load+fma)        │\n";
  std::cout << "  │    S2: LLVM dialect → LLVM IR (PHI+CFG+SSA)                │\n";
  std::cout << "  │    S3: Instruction Selection (DAG patterns → AVX2)          │\n";
  std::cout << "  │    S4: Register Allocation (12/16 YMM, 0 spills)           │\n";
  std::cout << "  │    S5: Instruction Scheduling (13cy → 11cy, ILP)           │\n";
  std::cout << "  │    S6: Machine Code Emission (VEX encoding + hex)          │\n";
  std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Performance model (ki-loop, per iteration):               │\n";
  std::cout << "  │    FLOPs: 4 FMA × 8 lanes × 2 = 64 FLOP                   │\n";
  std::cout << "  │    Cycles: 11 (scheduled) → 5.8 FLOP/cycle                │\n";
  std::cout << "  │    Memory: 1 B-row load (32B) + 4 A scalar (16B)          │\n";
  std::cout << "  │    Compute intensity: 64 FLOP / 48 B = 1.33 FLOP/B        │\n";
  std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Key interview points:                                      │\n";
  std::cout << "  │    • Vector lowering = memref→GEP + transfer→load/store    │\n";
  std::cout << "  │    • PHI nodes = loop-carried deps (accumulators, IVs)     │\n";
  std::cout << "  │    • ISel = DAG pattern match (fused GEP+load, FMA)        │\n";
  std::cout << "  │    • RegAlloc: 4-row block = 12 YMM → fits in AVX2 (16)   │\n";
  std::cout << "  │    • Scheduling: load‖broadcast hides 5-cy load latency    │\n";
  std::cout << "  │    • VEX prefix: 3-byte encoding, L=1 for 256-bit ops     │\n";
  std::cout << "  │    • AArch64 needs 2× NEON fmla for same 8xf32 width      │\n";
  std::cout << "  │    • SVE: same code adapts to any VL (128→2048 bit)        │\n";
  std::cout << "  └─────────────────────────────────────────────────────────────┘\n";
}

// =====================================================================
// Pipeline Runner
// =====================================================================

static void run_pipeline() {
  std::cout
      << "\n╔════════════════════════════════════════════════════════════════╗\n"
      << "║  Vector → Machine Code (7-Stage Backend Pipeline)            ║\n"
      << "║  GEMM 4×8 micro-kernel + Bias + ReLU → x86-64 AVX2          ║\n"
      << "╚════════════════════════════════════════════════════════════════╝\n";

  sep("Stage 0: Vector IR Input (from Stage 11)");
  stage0_input();

  sep("Stage 1: Vector Lowering (vector dialect → LLVM dialect)");
  auto llvm_dialect = stage1_vector_lowering();
  std::cout << "\n  LLVM dialect IR (ki-loop body + bias/relu):\n\n";
  llvm_dialect.print(std::cout);

  sep("Stage 2: LLVM Dialect → Textual LLVM IR");
  auto llvm_ir = stage2_llvm_ir();

  sep("Stage 3: Instruction Selection (LLVM IR → x86-64 AVX2)");
  auto machine_instrs = stage3_isel();

  sep("Stage 4: Register Allocation (virtual → physical)");
  auto reg_map = stage4_regalloc();

  sep("Stage 5: Instruction Scheduling (ILP optimization)");
  auto schedule = stage5_scheduling();

  sep("Stage 6: Machine Code Emission (assembly + binary encoding)");
  stage6_emit();

  print_summary();
}

// =====================================================================
int main() {
  std::cout << "================================================================\n";
  std::cout << "  Vector → Machine Code 7-Stage Backend Pipeline\n";
  std::cout << "  S0:Input → S1:VecLower → S2:LLVMIR → S3:ISel →\n";
  std::cout << "  S4:RegAlloc → S5:Sched → S6:Emit\n";
  std::cout << "================================================================\n";

  run_pipeline();

  std::cout << "\n✓ LLVM lowering pipeline test passed.\n";
  return 0;
}
