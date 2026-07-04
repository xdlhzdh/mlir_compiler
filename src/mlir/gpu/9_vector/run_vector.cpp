// run_vector.cpp — P9 (9_vector): Vector dialect pipeline (vector_ir, 6-step Pass chain)
//
// Takes the SCF/Affine output (scalar loops on memref) and lowers through
// the MLIR vector dialect to LLVM-ready vector intrinsics:
//
//   P9 Step 0: Precondition Check    — canonical loop check, stride analysis
//   P9 Step 1: Vectorization Prep    — alignment analysis, stride normalization
//   P9 Step 2: Vectorization Core    — scf.for → vector.transfer_read/write/contract
//   P9 Step 3: Vector Shaping        — unroll to SIMD width, register blocking, tail masking
//   P9 Step 4: Vector Optimization   — load/store coalescing, small-op fusion, redundant elim
//   P9 Step 5: Lowering to LLVM      — vector → llvm.intr.fmuladd / llvm.load / llvm.store
//
// Test case: GEMM micro-kernel + Bias + ReLU
//   Tiled matmul (32×8×32 tile), bias_add, relu — all on memref
//
// Pure C++17, header-only IR, no external dependencies.

#include "vector_ir.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace vec_ir;

// =====================================================================
// Helpers
// =====================================================================

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

// =====================================================================
// P9 Step 0: Vector Precondition Check
// =====================================================================
// Before vectorizing, verify:
//   (1) Loops are canonical (lb=0, step=1 or known constant)
//   (2) Memory access strides are statically known
//   (3) No complex control flow inside the loop body
// If any check fails, the loop is left scalar.

struct LoopInfo {
  std::string iv;
  int64_t lb, ub, step;
  bool canonical;
  bool stride_known;
  int stride;          // stride of the innermost access along this dim
  bool vectorizable;
};

static void stage0_precondition() {
  std::cout << "  Input: SCF/Affine output from Stage 10 (tiled GEMM micro-kernel)\n\n";

  std::vector<LoopInfo> loops = {
      {"%io", 0, 64, 32, true, true, 32, true},
      {"%jo", 0, 32, 8,  true, true, 8,  true},
      {"%ko", 0, 128, 32, true, true, 32, true},
      {"%ii", 0, 32, 1,  true, true, 1,  true},
      {"%ki", 0, 32, 1,  true, true, 1,  true},
      {"%fi", 0, 64, 1,  true, true, 1,  true},
      {"%fj", 0, 32, 8,  true, true, 1,  true},
  };

  std::cout << "  Loop precondition analysis:\n";
  std::cout << "    ┌──────────┬──────────┬──────────┬──────────┬───────────┐\n";
  std::cout << "    │ Loop     │ Canonic. │ Stride   │ Stride=? │ Vec OK?   │\n";
  std::cout << "    ├──────────┼──────────┼──────────┼──────────┼───────────┤\n";
  for (auto &l : loops) {
    std::printf("    │ %-8s │ %-8s │ %-8d │ %-8s │ %-9s │\n",
                l.iv.c_str(),
                l.canonical ? "yes" : "NO",
                l.stride,
                l.stride_known ? "known" : "UNKNOWN",
                l.vectorizable ? "yes" : "NO");
  }
  std::cout << "    └──────────┴──────────┴──────────┴──────────┴───────────┘\n\n";

  std::cout << "  Memory access analysis:\n";
  std::cout << "    A[%io+%ii, %ko+%ki]  → stride-1 along ki ✓ (contiguous)\n";
  std::cout << "    B[%ko+%ki, %jo+%ji]  → stride-1 along ji ✓ (contiguous)\n";
  std::cout << "    C[%io+%ii, %jo+%ji]  → stride-1 along ji ✓ (contiguous)\n";
  std::cout << "    bias[%fj]            → stride-1 ✓ (contiguous)\n";
  std::cout << "  → All loops pass precondition check. Ready to vectorize.\n";
}

// =====================================================================
// P9 Step 1: Vectorization Preparation
// =====================================================================
// Determine target SIMD width, check alignment, normalize strides.
//   - Target: vector<8xf32> (256-bit, AVX2 / NEON equiv.)
//   - Alignment: memref base is 32-byte aligned (assumed from alloc)
//   - The innermost vectorizable dimension has trip count 8 (or 32)
//     → perfect fit for vector<8xf32>, no tail needed at this step.

static void stage1_vec_prep() {
  std::cout << "  Target SIMD width: 8 × f32 (256-bit)\n\n";

  std::cout << "  Alignment analysis:\n";
  std::cout << "    %A (memref<64x128xf32>) : base 32-byte aligned ✓\n";
  std::cout << "      row stride = 128 × 4B = 512B (128-byte aligned ✓)\n";
  std::cout << "    %B (memref<128x32xf32>) : base 32-byte aligned ✓\n";
  std::cout << "      row stride = 32 × 4B = 128B (32-byte aligned ✓)\n";
  std::cout << "    %C (memref<64x32xf32>)  : base 32-byte aligned ✓\n";
  std::cout << "      row stride = 32 × 4B = 128B (32-byte aligned ✓)\n";
  std::cout << "    %bias (memref<32xf32>)  : base 32-byte aligned ✓\n\n";

  std::cout << "  Stride normalization:\n";
  std::cout << "    matmul vectorization target: ji-dimension (trip=8) → perfect\n";
  std::cout << "    bias+relu vectorization target: fj step 8 → perfect\n\n";

  std::cout << "  Loop shape stabilization:\n";
  std::cout << "    matmul tile: 32×8×32 → ii=32, ji→vector(8), ki=32 ✓\n";
  std::cout << "    bias+relu:   64×32   → fi=64, fj step 8 → 4 vector iters ✓\n";
  std::cout << "  → All dimensions stable. No dynamic trip counts.\n";
}

// =====================================================================
// P9 Step 2: Vectorization Core — scf.for → vector dialect
// =====================================================================
// Replace scalar load/arith/store loops with vector operations:
//
// GEMM micro-kernel (innermost ii→ki→ji body):
//   Before: scalar load A[ii,ki], load B[ki,ji], load C[ii,ji], mulf, addf, store
//   After:  vector.transfer_read C row → %vacc
//           loop ki: scalar load A → broadcast → vec<8>
//                    vector.transfer_read B row → vec<8>
//                    vector.fma %va, %vb, %vacc
//           vector.transfer_write %vacc → C row
//
// Bias+ReLU:
//   vector.transfer_read bias[fj:fj+8] → %vbias
//   vector.transfer_read C[fi,fj:fj+8] → %vc
//   arith.addf %vc, %vbias → %vc
//   arith.maxf %vc, %vzero → %vc
//   vector.transfer_write %vc → C[fi,fj:fj+8]

static VecFunc stage2_vec_core() {
  VecFunc f;
  f.name = "gemm_bias_relu_vec";
  f.args = {
      {"%A", MemRefType{{64, 128}, ElemType::F32}},
      {"%B", MemRefType{{128, 32}, ElemType::F32}},
      {"%bias", MemRefType{{32}, ElemType::F32}},
      {"%C", MemRefType{{64, 32}, ElemType::F32}},
  };

  VecType v8f32{{8}, ElemType::F32};

  // ── GEMM ──
  f.ops.push_back(VecOp::Comment("=== GEMM micro-kernel (tiled 32×8×32, vectorized) ==="));

  auto io = VecOp::Loop("%io", 0, 64, 32);
  auto jo = VecOp::Loop("%jo", 0, 32, 8);
  auto ko = VecOp::Loop("%ko", 0, 128, 32);

  auto ii = VecOp::Loop("%ii", 0, 32, 1);

  // Load C row vector
  ii.body.push_back(VecOp::TransferRead(
      "%vacc", "%C", {"%io+%ii", "%jo"}, v8f32, "C row → accumulator"));

  auto ki = VecOp::Loop("%ki", 0, 32, 1);
  ki.body.push_back(VecOp::TransferRead(
      "%a_scalar", "%A", {"%io+%ii", "%ko+%ki"}, {{1}, ElemType::F32},
      "scalar load A"));
  ki.body.push_back(VecOp::Broadcast("%va", "%a_scalar", v8f32));
  ki.body.push_back(VecOp::TransferRead(
      "%vb", "%B", {"%ko+%ki", "%jo"}, v8f32, "B row"));
  ki.body.push_back(VecOp::FMA("%vacc", "%va", "%vb", "%vacc", v8f32));

  ii.body.push_back(ki);
  ii.body.push_back(VecOp::TransferWrite(
      "%vacc", "%C", {"%io+%ii", "%jo"}, v8f32, "write back C row"));

  ko.body.push_back(ii);
  jo.body.push_back(ko);
  io.body.push_back(jo);
  f.ops.push_back(io);

  // ── vector.contract alternative (matmul tile as a single op) ──
  // In real MLIR, one can collapse the entire ki-reduction into a single
  // vector.contract when both A and B tiles fit in registers:
  //   %vA = vector.transfer_read A[io+ii, ko:ko+32]  : vector<32xf32>
  //   %vB = vector.transfer_read B[ko:ko+32, jo:jo+8] : vector<32x8xf32>
  //   %vacc = vector.contract %vA, %vB, %vacc
  //             {indexing_maps = [(k), (k,n) -> (n)],
  //              iterator_types = ["reduction", "parallel"]}
  //           : vector<32xf32>, vector<32x8xf32> → vector<8xf32>
  // This is semantically equivalent to the broadcast+fma loop above,
  // but expresses the contraction at a higher level for the optimizer.
  {
    VecType v32f32{{32}, ElemType::F32};
    VecType v32x8f32{{32, 8}, ElemType::F32};

    f.ops.push_back(VecOp::Comment(
        "=== Alternative: vector.contract (single-op matmul tile) ==="));

    auto contract_io = VecOp::Loop("%cio", 0, 64, 32);
    auto contract_jo = VecOp::Loop("%cjo", 0, 32, 8);
    auto contract_ko = VecOp::Loop("%cko", 0, 128, 32);
    auto contract_ii = VecOp::Loop("%cii", 0, 32, 1);

    contract_ii.body.push_back(VecOp::TransferRead(
        "%vacc", "%C", {"%cio+%cii", "%cjo"}, v8f32, "C row accumulator"));
    contract_ii.body.push_back(VecOp::TransferRead(
        "%vA_row", "%A", {"%cio+%cii", "%cko"}, v32f32,
        "A row tile [1×32]"));
    contract_ii.body.push_back(VecOp::TransferRead(
        "%vB_tile", "%B", {"%cko", "%cjo"}, v32x8f32,
        "B sub-tile [32×8]"));
    contract_ii.body.push_back(VecOp::Contract(
        "%vacc", "%vA_row", "%vB_tile", "%vacc", v8f32,
        "k-reduction: (k)×(k,n)→(n), eliminates ki loop"));
    contract_ii.body.push_back(VecOp::TransferWrite(
        "%vacc", "%C", {"%cio+%cii", "%cjo"}, v8f32, "write back C row"));

    contract_ko.body.push_back(contract_ii);
    contract_jo.body.push_back(contract_ko);
    contract_io.body.push_back(contract_jo);
    f.ops.push_back(contract_io);

    std::cout << "  [contract] Added vector.contract alternative for GEMM tile:\n";
    std::cout << "    vector.contract %vA_row(32), %vB_tile(32×8), %vacc(8)\n";
    std::cout << "    → replaces the entire ki-loop (broadcast+fma) with 1 op\n";
    std::cout << "    → optimizer can lower to fma-chain or HW matmul unit\n\n";
  }

  // ── Bias + ReLU ──
  f.ops.push_back(VecOp::Comment("=== Bias + ReLU (fused, vectorized) ==="));
  f.ops.push_back(VecOp::Splat("%vzero", "0.0", v8f32));

  auto fi = VecOp::Loop("%fi", 0, 64, 1);
  auto fj = VecOp::Loop("%fj", 0, 32, 8);

  fj.body.push_back(VecOp::TransferRead(
      "%vbias", "%bias", {"%fj"}, v8f32, "bias vector"));
  fj.body.push_back(VecOp::TransferRead(
      "%vc", "%C", {"%fi", "%fj"}, v8f32, "C row"));
  fj.body.push_back(VecOp::Arith("arith.addf", "%vc", "%vc", "%vbias", v8f32));
  fj.body.push_back(VecOp::Arith("arith.maxf", "%vc", "%vc", "%vzero", v8f32));
  fj.body.push_back(VecOp::TransferWrite(
      "%vc", "%C", {"%fi", "%fj"}, v8f32, "write back C row"));

  fi.body.push_back(fj);
  f.ops.push_back(fi);

  int reads = f.count(OpKind::TRANSFER_READ);
  int writes = f.count(OpKind::TRANSFER_WRITE);
  int fmas = f.count(OpKind::FMA);
  int contracts = f.count(OpKind::CONTRACT);
  std::cout << "  Scalar → Vector conversion:\n";
  std::cout << "    vector.transfer_read  : " << reads << "\n";
  std::cout << "    vector.transfer_write : " << writes << "\n";
  std::cout << "    vector.fma            : " << fmas << "\n";
  std::cout << "    vector.contract       : " << contracts << "\n";
  std::cout << "    vector.broadcast      : " << f.count(OpKind::BROADCAST) << "\n";
  std::cout << "    arith (vector)        : " << f.count(OpKind::ARITH) << "\n";
  std::cout << "    All innermost ops now on vector<8xf32>.\n";
  return f;
}

// =====================================================================
// P9 Step 3: Vector Shaping — unroll, register blocking, tail masking
// =====================================================================
// (a) Register blocking: unroll ii by 4 → 4 accumulators (%vacc0..3)
//     Benefit: hide FMA latency, keep FPU pipeline full.
//
// (b) Tail handling: N=32 is divisible by 8, no tail for the main case.
//     Demo a 30-element case: 3 full vectors (24) + 1 masked vector (6/8).

static VecFunc stage3_shaping(VecFunc f) {
  VecType v8f32{{8}, ElemType::F32};

  // (a) show register blocking concept
  std::cout << "  (a) Register blocking: unroll ii by 4\n";
  std::cout << "      Before: 1 accumulator per ii iteration\n";
  std::cout << "      After:  4 accumulators (%vacc0..%vacc3)\n\n";

  std::cout << "      scf.for %ii = 0 to 32 step 4 {\n";
  std::cout << "        %vacc0 = vector.transfer_read C[ii+0, jo] : vector<8xf32>\n";
  std::cout << "        %vacc1 = vector.transfer_read C[ii+1, jo] : vector<8xf32>\n";
  std::cout << "        %vacc2 = vector.transfer_read C[ii+2, jo] : vector<8xf32>\n";
  std::cout << "        %vacc3 = vector.transfer_read C[ii+3, jo] : vector<8xf32>\n";
  std::cout << "        scf.for %ki = 0 to 32 {\n";
  std::cout << "          %vb = vector.transfer_read B[ki, jo] : vector<8xf32>\n";
  std::cout << "          %va0 = broadcast A[ii+0, ki]; %vacc0 = fma va0, vb, vacc0\n";
  std::cout << "          %va1 = broadcast A[ii+1, ki]; %vacc1 = fma va1, vb, vacc1\n";
  std::cout << "          %va2 = broadcast A[ii+2, ki]; %vacc2 = fma va2, vb, vacc2\n";
  std::cout << "          %va3 = broadcast A[ii+3, ki]; %vacc3 = fma va3, vb, vacc3\n";
  std::cout << "        }  // B loaded once, shared by 4 rows → 4× data reuse\n";
  std::cout << "        vector.transfer_write %vacc0..%vacc3 → C\n";
  std::cout << "      }\n\n";
  std::cout << "      Benefit: 4 independent FMA chains → hide FMA latency (4~5 cy)\n";
  std::cout << "      Registers used: 4 acc + 1 B + 4 A broadcasts = 9 of 16 YMM regs\n\n";

  // (b) Tail masking demo
  std::cout << "  (b) Tail handling (example: N=30 instead of N=32)\n";
  std::cout << "      Full vectors: 30/8 = 3 full iters (24 elements)\n";
  std::cout << "      Tail: 30-24 = 6 elements → masked vector\n\n";

  // Build tail masking example as separate ops
  f.ops.push_back(VecOp::Comment("=== Tail masking demo (N=30, tail=6) ==="));
  f.ops.push_back(VecOp::CreateMask("%mask", 8, 6));
  f.ops.push_back(VecOp::MaskedLoad(
      "%vtail", "%C", {"%fi", "24"}, "%mask", v8f32));
  f.ops.push_back(VecOp::Arith("arith.addf", "%vtail", "%vtail", "%vbias_tail", v8f32));
  f.ops.push_back(VecOp::Arith("arith.maxf", "%vtail", "%vtail", "%vzero", v8f32));
  f.ops.push_back(VecOp::MaskedStore(
      "%vtail", "%C", {"%fi", "24"}, "%mask", v8f32));

  std::cout << "      vector.create_mask 6 : vector<8xi1>  (lanes 0~5 active, 6~7 off)\n";
  std::cout << "      vector.maskedload  C[fi, 24], %mask → %vtail\n";
  std::cout << "      arith on %vtail (inactive lanes untouched)\n";
  std::cout << "      vector.maskedstore %vtail, C[fi, 24], %mask\n";
  std::cout << "      → no remainder scalar loop needed!\n";

  return f;
}

// =====================================================================
// P9 Step 4: Vector Optimization
// =====================================================================
// (a) Load/store coalescing: consecutive transfer_read from same memref
//     at offset 0 and offset 8 → single wider read (conceptual).
//     In MLIR: keep vector<8xf32> but ensure no redundant loads.
//
// (b) Vector fusion: back-to-back arith.addf + arith.maxf on same register
//     → fused into one pass (both in same scheduling slot).
//
// (c) Redundant vector elimination: if %vacc = transfer_read then
//     immediately transfer_write of same %vacc with no modification → dead.

static VecFunc stage4_vec_opt(VecFunc f) {
  std::cout << "  (a) Load/store coalescing:\n";
  std::cout << "      B[ki, jo:jo+8] loaded once per ki iteration,\n";
  std::cout << "      shared across 4 unrolled ii rows (from Stage 3)\n";
  std::cout << "      → 1 load serves 4 FMAs (4× reuse)\n\n";

  std::cout << "  (b) Vector fusion (small-op merging):\n";
  std::cout << "      bias_add + relu:\n";
  std::cout << "        %vc = addf %vc, %vbias → %vc = maxf %vc, %vzero\n";
  std::cout << "        Back-to-back on same register → no pipeline stall\n";
  std::cout << "        Scheduler: addf latency hidden by maxf issue\n\n";

  std::cout << "  (c) Redundant vector elimination:\n";
  std::cout << "      Removed: dead transfer_read/write pair with no use\n";
  std::cout << "      (e.g., debug loads that were leftover)\n\n";

  // Remove tail masking demo ops for the "clean" final output
  while (!f.ops.empty() && f.ops.back().kind != OpKind::LOOP &&
         f.ops.back().kind != OpKind::SPLAT) {
    f.ops.pop_back();
  }

  std::cout << "  Optimization summary:\n";
  std::cout << "    B-tile loads coalesced: 4× reuse per ki step\n";
  std::cout << "    Fused bias+relu:        addf+maxf back-to-back\n";
  std::cout << "    Dead code removed:      0 redundant transfer ops\n";
  return f;
}

// =====================================================================
// P9 Step 5: Lowering to LLVM / GPU intrinsics
// =====================================================================
// vector.transfer_read  → llvm.load (aligned)
// vector.transfer_write → llvm.store (aligned)
// vector.fma            → llvm.intr.fmuladd (x86: _mm256_fmadd_ps)
// vector.broadcast      → llvm.intr.broadcast / shufflevector
// arith.addf on vector  → llvm.fadd <8 x float>
// arith.maxf on vector  → llvm.intr.maxnum / _mm256_max_ps
// vector.maskedload     → llvm.masked.load
// vector.maskedstore    → llvm.masked.store

static VecFunc stage5_llvm_lower(VecFunc f) {
  VecType v8f32{{8}, ElemType::F32};

  VecFunc llvm_f;
  llvm_f.name = "gemm_bias_relu_llvm";
  llvm_f.args = f.args;

  llvm_f.ops.push_back(VecOp::Comment(
      "=== LLVM IR (x86-64 AVX2 target) ==="));

  // GEMM micro-kernel in LLVM form
  llvm_f.ops.push_back(VecOp::Comment("GEMM outer: parallel on io tiles"));

  auto io = VecOp::Loop("%io", 0, 64, 32);
  auto jo = VecOp::Loop("%jo", 0, 32, 8);
  auto ko = VecOp::Loop("%ko", 0, 128, 32);
  auto ii = VecOp::Loop("%ii", 0, 32, 4);

  ii.body.push_back(VecOp::Comment("register-blocked: 4 rows × vector<8xf32>"));
  ii.body.push_back(VecOp::LLVMLoad("%vacc0", "%C_ptr[ii+0,jo]", v8f32));
  ii.body.push_back(VecOp::LLVMLoad("%vacc1", "%C_ptr[ii+1,jo]", v8f32));
  ii.body.push_back(VecOp::LLVMLoad("%vacc2", "%C_ptr[ii+2,jo]", v8f32));
  ii.body.push_back(VecOp::LLVMLoad("%vacc3", "%C_ptr[ii+3,jo]", v8f32));

  auto ki = VecOp::Loop("%ki", 0, 32, 1);
  ki.body.push_back(VecOp::LLVMLoad("%vb", "%B_ptr[ki,jo]", v8f32));
  ki.body.push_back(VecOp::LLVMIntrin(
      "llvm.intr.fmuladd", "%vacc0", "%va0_bcast", "%vb", "%vacc0", v8f32));
  ki.body.push_back(VecOp::LLVMIntrin(
      "llvm.intr.fmuladd", "%vacc1", "%va1_bcast", "%vb", "%vacc1", v8f32));
  ki.body.push_back(VecOp::LLVMIntrin(
      "llvm.intr.fmuladd", "%vacc2", "%va2_bcast", "%vb", "%vacc2", v8f32));
  ki.body.push_back(VecOp::LLVMIntrin(
      "llvm.intr.fmuladd", "%vacc3", "%va3_bcast", "%vb", "%vacc3", v8f32));

  ii.body.push_back(ki);
  ii.body.push_back(VecOp::LLVMStore("%vacc0", "%C_ptr[ii+0,jo]", v8f32));
  ii.body.push_back(VecOp::LLVMStore("%vacc1", "%C_ptr[ii+1,jo]", v8f32));
  ii.body.push_back(VecOp::LLVMStore("%vacc2", "%C_ptr[ii+2,jo]", v8f32));
  ii.body.push_back(VecOp::LLVMStore("%vacc3", "%C_ptr[ii+3,jo]", v8f32));

  ko.body.push_back(ii);
  jo.body.push_back(ko);
  io.body.push_back(jo);
  llvm_f.ops.push_back(io);

  // Bias + ReLU in LLVM form
  llvm_f.ops.push_back(VecOp::Comment("bias + relu (fused, LLVM IR)"));

  auto fi = VecOp::Loop("%fi", 0, 64, 1);
  auto fj = VecOp::Loop("%fj", 0, 32, 8);
  fj.body.push_back(VecOp::LLVMLoad("%vbias", "%bias_ptr[fj]", v8f32));
  fj.body.push_back(VecOp::LLVMLoad("%vc", "%C_ptr[fi,fj]", v8f32));
  fj.body.push_back(VecOp::LLVMIntrin(
      "llvm.fadd", "%vc", "%vc", "%vbias", "%_unused", v8f32));
  fj.body.push_back(VecOp::LLVMIntrin(
      "llvm.intr.maxnum", "%vc", "%vc", "%vzero", "%_unused", v8f32));
  fj.body.push_back(VecOp::LLVMStore("%vc", "%C_ptr[fi,fj]", v8f32));
  fi.body.push_back(fj);
  llvm_f.ops.push_back(fi);

  std::cout << "  Vector → LLVM IR lowering map:\n";
  std::cout << "    ┌──────────────────────────┬─────────────────────────────────┐\n";
  std::cout << "    │ Vector Dialect            │ LLVM IR / Intrinsic             │\n";
  std::cout << "    ├──────────────────────────┼─────────────────────────────────┤\n";
  std::cout << "    │ vector.transfer_read      │ llvm.load <8 x float>*, align32│\n";
  std::cout << "    │ vector.transfer_write     │ llvm.store <8 x float>, align32│\n";
  std::cout << "    │ vector.fma                │ llvm.intr.fmuladd              │\n";
  std::cout << "    │ vector.broadcast          │ shufflevector (splat)          │\n";
  std::cout << "    │ arith.addf (vec)          │ fadd <8 x float>              │\n";
  std::cout << "    │ arith.maxf (vec)          │ llvm.intr.maxnum               │\n";
  std::cout << "    │ vector.maskedload         │ llvm.masked.load               │\n";
  std::cout << "    │ vector.maskedstore        │ llvm.masked.store              │\n";
  std::cout << "    └──────────────────────────┴─────────────────────────────────┘\n\n";

  std::cout << "  x86-64 AVX2 ISA mapping:\n";
  std::cout << "    llvm.intr.fmuladd → vfmadd231ps (%ymm)\n";
  std::cout << "    llvm.load align32 → vmovaps (%ymm)\n";
  std::cout << "    llvm.intr.maxnum  → vmaxps (%ymm)\n";
  std::cout << "    shufflevector     → vbroadcastss\n\n";

  std::cout << "  AArch64 NEON/SVE mapping:\n";
  std::cout << "    llvm.intr.fmuladd → fmla v0.4s (NEON) / fmla z0.s (SVE)\n";
  std::cout << "    llvm.load         → ld1 {v0.4s}  (NEON 128-bit)\n";
  std::cout << "    llvm.intr.maxnum  → fmaxnm v0.4s\n";

  return llvm_f;
}

// =====================================================================
// Pipeline Summary
// =====================================================================

static void print_summary() {
  std::cout << "\n  ┌────────────────────────────────────────────────────────┐\n";
  std::cout << "  │            Vector Pipeline Summary                     │\n";
  std::cout << "  ├────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Input : SCF loops with scalar memref.load/store      │\n";
  std::cout << "  │  Output: LLVM IR with <8 x float> intrinsics          │\n";
  std::cout << "  ├────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Optimizations applied:                                │\n";
  std::cout << "  │    ✓ Precondition check (canonical + stride)  (S0)     │\n";
  std::cout << "  │    ✓ Alignment + stride analysis              (S1)     │\n";
  std::cout << "  │    ✓ Scalar → vector dialect (transfer_read/  (S2)     │\n";
  std::cout << "  │      write, fma, broadcast, contract)                  │\n";
  std::cout << "  │    ✓ Register blocking (4 rows × vec<8>)      (S3)     │\n";
  std::cout << "  │    ✓ Tail masking (vector.create_mask)         (S3)     │\n";
  std::cout << "  │    ✓ Load coalescing + small-op fusion         (S4)     │\n";
  std::cout << "  │    ✓ vector → llvm.intr lowering               (S5)     │\n";
  std::cout << "  ├────────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Key interview points:                                 │\n";
  std::cout << "  │    • transfer_read/write = safe vector load/store      │\n";
  std::cout << "  │      (handles OOB with padding, masking available)     │\n";
  std::cout << "  │    • vector.contract = generalized matmul (dot prod)   │\n";
  std::cout << "  │    • Register blocking = hide FMA latency (4-5 cy)     │\n";
  std::cout << "  │    • Tail masking = avoid scalar remainder loops       │\n";
  std::cout << "  │    • AVX2: vfmadd231ps = 8 FMA/cy, vmovaps = 32B/cy   │\n";
  std::cout << "  │    • Peak: 16 FLOP/cy (1 FMA = 2 FLOP × 8 lanes)     │\n";
  std::cout << "  └────────────────────────────────────────────────────────┘\n";
}

// =====================================================================
// Pipeline Runner
// =====================================================================

static void run_pipeline() {
  std::cout
      << "\n╔═══════════════════════════════════════════════════════════════╗\n"
      << "║  GEMM + Bias + ReLU (Vector Dialect 6-Stage Pipeline)       ║\n"
      << "║  Tiled matmul 32×8×32 + bias_add + relu → LLVM IR          ║\n"
      << "╚═══════════════════════════════════════════════════════════════╝\n";

  // ── P9 Step 0 ──
  sep("Stage 0: Vector Precondition Check");
  stage0_precondition();

  // ── P9 Step 1 ──
  sep("Stage 1: Vectorization Preparation (alignment + stride)");
  stage1_vec_prep();

  // ── P9 Step 2 ──
  sep("Stage 2: Vectorization Core (scf → vector dialect)");
  auto f = stage2_vec_core();
  std::cout << "\n";
  f.print(std::cout);

  // ── P9 Step 3 ──
  sep("Stage 3: Vector Shaping (register blocking + tail masking)");
  f = stage3_shaping(std::move(f));
  std::cout << "\n  Shaped IR (with tail masking demo appended):\n\n";
  f.print(std::cout);

  // ── P9 Step 4 ──
  sep("Stage 4: Vector Optimization (coalesce + fuse + eliminate)");
  f = stage4_vec_opt(std::move(f));

  // ── P9 Step 5 ──
  sep("Stage 5: Lowering to LLVM / GPU intrinsics");
  auto llvm_f = stage5_llvm_lower(std::move(f));
  std::cout << "\n  Final LLVM IR (pseudo):\n\n";
  llvm_f.print(std::cout);

  // ── Summary ──
  print_summary();
}

// =====================================================================
int main() {
  std::cout << "================================================================\n";
  std::cout << "  Vector Dialect 6-Stage Pipeline\n";
  std::cout << "  S0:Precondition → S1:Prep → S2:Core → S3:Shaping →\n";
  std::cout << "  S4:Optimize → S5:LLVM\n";
  std::cout << "================================================================\n";

  run_pipeline();

  std::cout << "\n✓ Vector pipeline test passed.\n";
  return 0;
}
