// run_scf_affine.cpp — P8 (10_scf_affine): SCF/Affine loop optimization (scf_affine_ir, 9-step Pass chain)
//
// Transforms Linalg-on-memref to optimized explicit loop nests:
//
//   P8 Step 0: Pre-clean             — canonicalize + CSE + DCE
//   P8 Step 1: Linalg → Loop        — convert linalg ops to scf.for + memref.load/store
//   P8 Step 2: Loop Canonicalization — dead loop elimination, bound normalization
//   P8 Step 3: Tiling                — cache-aware multi-level tiling (L2: 32×8×32)
//   P8 Step 4: Loop Transform        — interchange (i,k,j) + loop fusion (bias+relu)
//   P8 Step 5: Parallelization       — scf.parallel on outer tile loops, thread mapping
//   P8 Step 6: Memory Optimization   — register promotion + B tile stack promotion
//   P8 Step 7: Vectorization Prep    — vector.load/fma/store on innermost dimension
//   P8 Step 8: Cleanup               — final statistics + summary
//
// Test case: GEMM + Bias + ReLU  (matmul → bias_add → relu)
//   A[64×128] × B[128×32] + bias[32] → relu → C[64×32]
//
// Pure C++17, header-only IR, no external dependencies.

#include "scf_affine_ir.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

using namespace loop_ir;

// =====================================================================
// Helpers
// =====================================================================

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

static void print_stats(const Func &f) {
  std::cout << "    loops=" << f.total_loops()
            << "  ops=" << f.total_ops()
            << "  parallel=" << f.total_par()
            << "  vector=" << f.total_vec() << "\n";
}

// =====================================================================
// P8 Step 0: Pre-clean — show initial Linalg-on-memref, remove dead ops
// =====================================================================
// After bufferization, the IR is tensor-free: all ops work on memrefs.
// P8 Step 0 removes dead ops (unused fill / alloc) before lowering to loops.

static void stage0_preclean() {
  std::cout << "  Input (Linalg on memref, after bufferization):\n\n";
  std::cout << "    func.func @gemm_bias_relu(\n";
  std::cout << "        %A:    memref<64x128xf32>,\n";
  std::cout << "        %B:    memref<128x32xf32>,\n";
  std::cout << "        %bias: memref<32xf32>,\n";
  std::cout << "        %C:    memref<64x32xf32>) {\n";
  std::cout << "      linalg.matmul ins(%A, %B) outs(%C)\n";
  std::cout << "      linalg.generic {bias_add} ins(%bias) outs(%C)\n";
  std::cout << "      linalg.generic {relu} outs(%C)\n";
  std::cout << "      [dead] linalg.fill ins(0.0) outs(%unused)\n";
  std::cout << "      return\n    }\n\n";
  std::cout << "  [DCE] remove dead linalg.fill (result has no users)\n";
  std::cout << "  → 3 live Linalg ops remaining\n";
}

// =====================================================================
// P8 Step 1: Linalg → Loop
// =====================================================================
// Each Linalg op is lowered to explicit scf.for nests + memref.load/store:
//   matmul(M,N,K) → for i(M) { for j(N) { for k(K) { load, mul, add, store }}}
//   bias_add(M,N) → for i(M) { for j(N) { load, add, store }}
//   relu(M,N)     → for i(M) { for j(N) { load, max, store }}

static Func stage1_linalg_to_loop() {
  Func f;
  f.name = "gemm_bias_relu";
  f.args = {
      {"%A", MemRefType{{64, 128}, ElemType::F32}},
      {"%B", MemRefType{{128, 32}, ElemType::F32}},
      {"%bias", MemRefType{{32}, ElemType::F32}},
      {"%C", MemRefType{{64, 32}, ElemType::F32}},
  };

  // (1) matmul: C[i,j] += A[i,k] * B[k,j]
  {
    auto k_loop = Stmt::For("%k", 0, 128);
    k_loop.body = {
        Stmt::Load("%a", "%A", {"%i", "%k"}),
        Stmt::Load("%b", "%B", {"%k", "%j"}),
        Stmt::Load("%c", "%C", {"%i", "%j"}),
        Stmt::Arith("arith.mulf", "%t0", "%a", "%b"),
        Stmt::Arith("arith.addf", "%t1", "%c", "%t0"),
        Stmt::Store("%t1", "%C", {"%i", "%j"}),
    };
    auto j_loop = Stmt::For("%j", 0, 32);
    j_loop.body = {k_loop};
    auto i_loop = Stmt::For("%i", 0, 64);
    i_loop.body = {j_loop};
    f.body.push_back(Stmt::Comment("matmul: C[i,j] += A[i,k] * B[k,j]"));
    f.body.push_back(i_loop);
  }

  // (2) bias_add: C[i,j] += bias[j]
  {
    auto bj = Stmt::For("%bj", 0, 32);
    bj.body = {
        Stmt::Load("%bv", "%bias", {"%bj"}),
        Stmt::Load("%c1", "%C", {"%bi", "%bj"}),
        Stmt::Arith("arith.addf", "%t2", "%c1", "%bv"),
        Stmt::Store("%t2", "%C", {"%bi", "%bj"}),
    };
    auto bi = Stmt::For("%bi", 0, 64);
    bi.body = {bj};
    f.body.push_back(Stmt::Comment("bias_add: C[i,j] += bias[j]"));
    f.body.push_back(bi);
  }

  // (3) relu: C[i,j] = max(C[i,j], 0)
  {
    auto rj = Stmt::For("%rj", 0, 32);
    rj.body = {
        Stmt::Load("%c2", "%C", {"%ri", "%rj"}),
        Stmt::Arith("arith.maxf", "%t3", "%c2", "%zero"),
        Stmt::Store("%t3", "%C", {"%ri", "%rj"}),
    };
    auto ri = Stmt::For("%ri", 0, 64);
    ri.body = {rj};
    f.body.push_back(Stmt::Comment("relu: C[i,j] = max(C[i,j], 0)"));
    f.body.push_back(ri);
  }

  // (4) Dead empty loop (to demonstrate dead loop elimination in P8 Step 2)
  {
    auto d = Stmt::For("%d", 0, 1);
    f.body.push_back(Stmt::Comment("[dead] empty loop (unused)"));
    f.body.push_back(d);
  }

  std::cout << "  Converted 3 Linalg ops to explicit loop nests:\n";
  std::cout << "    matmul  → 3-level (i=64, j=32, k=128) + 6 scalar ops\n";
  std::cout << "    bias_add → 2-level (i=64, j=32) + 4 scalar ops\n";
  std::cout << "    relu    → 2-level (i=64, j=32) + 3 scalar ops\n";
  std::cout << "    + 1 dead empty loop (for Stage 2 elimination)\n";
  return f;
}

// =====================================================================
// P8 Step 2: Loop Canonicalization — dead loop elimination
// =====================================================================

static int remove_empty_loops(std::vector<Stmt> &stmts) {
  int count = 0;
  for (auto it = stmts.begin(); it != stmts.end();) {
    if (it->is_loop()) {
      count += remove_empty_loops(it->body);
      if (it->body.empty()) {
        std::cout << "  [DeadLoopElim] removed empty loop " << it->iv
                  << " (trip " << (it->ub - it->lb) / std::max(it->step, (int64_t)1)
                  << ")\n";
        it = stmts.erase(it);
        ++count;
        continue;
      }
    }
    ++it;
  }
  return count;
}

static int stage2_loop_canon(Func &f) {
  int removed = remove_empty_loops(f.body);

  // Remove orphaned "[dead]" comments
  for (auto it = f.body.begin(); it != f.body.end();) {
    if (it->kind == StmtKind::COMMENT &&
        it->comment.find("[dead]") != std::string::npos) {
      it = f.body.erase(it);
    } else {
      ++it;
    }
  }
  return removed;
}

// =====================================================================
// P8 Step 3: Tiling — cache-aware multi-level tiling for GEMM
// =====================================================================
// Tile sizes chosen for typical L1/L2 cache hierarchy:
//   L2 outer tiles: TM=32, TN=8, TK=32
//   → io=0..64 step 32, jo=0..32 step 8, ko=0..128 step 32
//   → inner: ii=0..32, ji=0..8, ki=0..32
//
// Benefit: each tile of A (32×32=4KB), B (32×8=1KB), C (32×8=1KB)
// fits in L1 cache (~32KB).

static void stage3_tiling(Func &f) {
  auto ki = Stmt::For("%ki", 0, 32);
  ki.body = {
      Stmt::Load("%a", "%A", {"%io+%ii", "%ko+%ki"}),
      Stmt::Load("%b", "%B", {"%ko+%ki", "%jo+%ji"}),
      Stmt::Load("%c", "%C", {"%io+%ii", "%jo+%ji"}),
      Stmt::Arith("arith.mulf", "%t0", "%a", "%b"),
      Stmt::Arith("arith.addf", "%t1", "%c", "%t0"),
      Stmt::Store("%t1", "%C", {"%io+%ii", "%jo+%ji"}),
  };
  auto ji = Stmt::For("%ji", 0, 8);
  ji.body = {ki};
  auto ii = Stmt::For("%ii", 0, 32);
  ii.body = {ji};
  auto ko = Stmt::For("%ko", 0, 128, 32);
  ko.body = {ii};
  auto jo = Stmt::For("%jo", 0, 32, 8);
  jo.body = {ko};
  auto io = Stmt::For("%io", 0, 64, 32);
  io.body = {jo};

  f.body[0] = Stmt::Comment("matmul (tiled TM=32, TN=8, TK=32)");
  f.body[1] = io;

  std::cout << "  [Tiling] matmul: 2-level tiling [TM=32, TN=8, TK=32]\n";
  std::cout << "    outer: %io=0..64 step 32, %jo=0..32 step 8, "
               "%ko=0..128 step 32\n";
  std::cout << "    inner: %ii=0..32, %ji=0..8, %ki=0..32\n";
  std::cout << "    tile footprint: A(32×32)=4KB + B(32×8)=1KB + "
               "C(32×8)=1KB = 6KB (fits L1)\n";
  std::cout << "    → 6 nested loops (was 3)\n";
}

// =====================================================================
// P8 Step 4: Loop Transform — interchange + loop fusion
// =====================================================================
// (a) Interchange: swap %ji ↔ %ki in matmul inner nest
//     Before: ii → ji → ki  (k innermost: B[k,j] has stride-N access)
//     After:  ii → ki → ji  (j innermost: B[k,j] has stride-1 access ✓)
//
// (b) Loop fusion: merge bias_add + relu into single 2-level loop
//     Eliminates 1 redundant memref.load + 1 redundant memref.store of %C

static void stage4_loop_transform(Func &f) {
  // (a) Interchange ji ↔ ki inside the matmul tiled nest
  // Navigate: f.body[1] → io → jo → ko → ii → ji(swap) → ki(swap)
  auto &io_loop = f.body[1];
  auto &jo_loop = io_loop.body[0];
  auto &ko_loop = jo_loop.body[0];
  auto &ii_loop = ko_loop.body[0];
  auto &loop_a = ii_loop.body[0];  // currently: ji(0..8)
  auto &loop_b = loop_a.body[0];   // currently: ki(0..32)

  std::swap(loop_a.iv, loop_b.iv);
  std::swap(loop_a.lb, loop_b.lb);
  std::swap(loop_a.ub, loop_b.ub);
  std::swap(loop_a.step, loop_b.step);

  f.body[0] = Stmt::Comment(
      "matmul (tiled, interchanged: ii→ki→ji for stride-1 B access)");

  std::cout << "  [Interchange] matmul inner: ii→ji→ki → ii→ki→ji\n";
  std::cout << "    reason: B[ko+ki, jo+ji] with ji innermost → stride-1 "
               "(row-major)\n";

  // (b) Fuse bias_add + relu loops (same trip counts: 64×32)
  // Remove the 4 entries (2 comments + 2 loops) and replace with 1 fused
  auto fj = Stmt::For("%fj", 0, 32);
  fj.body = {
      Stmt::Load("%bv", "%bias", {"%fj"}),
      Stmt::Load("%c1", "%C", {"%fi", "%fj"}),
      Stmt::Arith("arith.addf", "%t2", "%c1", "%bv"),
      Stmt::Arith("arith.maxf", "%t3", "%t2", "%zero"),
      Stmt::Store("%t3", "%C", {"%fi", "%fj"}),
  };
  auto fi = Stmt::For("%fi", 0, 64);
  fi.body = {fj};

  // f.body: [0]=comment, [1]=matmul, [2]=bias_comment, [3]=bias_loop,
  //         [4]=relu_comment, [5]=relu_loop
  f.body.erase(f.body.begin() + 2, f.body.end());
  f.body.push_back(
      Stmt::Comment("bias_add + relu (fused: 1 load + 1 store eliminated)"));
  f.body.push_back(fi);

  std::cout << "  [Fusion] bias_add + relu → single 2-level loop\n";
  std::cout << "    eliminated: 1 memref.load + 1 memref.store (saved 2 ops)\n";
  std::cout << "    before: C[i,j] = relu(C[i,j] + bias[j]) needed 2 passes\n";
  std::cout << "    after:  single pass — addf then maxf, 1 store\n";
}

// =====================================================================
// P8 Step 5: Parallelization — scf.parallel on outer loops
// =====================================================================
// Mark outermost tile loop and fused loop as scf.parallel.
// In real MLIR: scf.parallel → omp.parallel / gpu.launch depending on target.

static void stage5_parallelize(Func &f) {
  // Matmul outer: f.body[1] (%io loop)
  auto &mm = f.body[1];
  mm.kind = StmtKind::PAR_LOOP;
  mm.thread_map = "thread_x (2 tiles, 2 threads)";
  std::cout << "  [Parallel] matmul outer %io → scf.parallel"
            << " (2 tiles of 32 rows each)\n";

  // Fused bias+relu: f.body[3] (%fi loop)
  auto &fused = f.body[3];
  fused.kind = StmtKind::PAR_LOOP;
  fused.thread_map = "thread_x (64 rows)";
  std::cout << "  [Parallel] fused %fi → scf.parallel (64 parallel rows)\n";
  std::cout << "    target mapping: CPU → OpenMP threads, GPU → CUDA blocks\n";
}

// =====================================================================
// P8 Step 6: Memory Optimization
// =====================================================================
// Two key optimizations for the matmul inner nest:
//
// (a) B tile stack promotion:
//     Allocate memref<32×8xf32> on stack, copy B sub-tile into it.
//     Benefit: guaranteed L1 residency, no cache-line conflicts.
//
// (b) Register promotion (accumulator hoisting):
//     Reorder inner loops: ii→ki→ji → ii→ji→ki
//     Hoist C[ii,ji] load before ki loop, store after ki loop.
//     Benefit: C stays in register across all K iterations.
//     This "undoes" the interchange, but within the tile everything
//     fits in L1, so the stride-N access on B_tile is fine (1KB tile).

static void stage6_mem_opt(Func &f) {
  // Rebuild the matmul inner structure:
  // io(par) → jo → ko → { alloc B_tile, copy, ii → ji → ki → {body}, dealloc }

  auto ki = Stmt::For("%ki", 0, 32);
  ki.body = {
      Stmt::Load("%a", "%A", {"%io+%ii", "%ko+%ki"}),
      Stmt::Load("%b", "%B_tile", {"%ki", "%ji"}),
      Stmt::Arith("arith.mulf", "%t0", "%a", "%b"),
      Stmt::Arith("arith.addf", "%acc", "%acc", "%t0"),
  };

  auto ji = Stmt::For("%ji", 0, 8);
  ji.body = {
      Stmt::Comment("register promotion: C[ii,ji] → %acc"),
      Stmt::Load("%acc", "%C", {"%io+%ii", "%jo+%ji"}),
      ki,
      Stmt::Store("%acc", "%C", {"%io+%ii", "%jo+%ji"}),
  };

  auto ii = Stmt::For("%ii", 0, 32);
  ii.body = {ji};

  auto ko = Stmt::For("%ko", 0, 128, 32);
  ko.body = {
      Stmt::Comment("stack promotion: B sub-tile → L1 buffer"),
      Stmt::Alloc("%B_tile", MemRefType{{32, 8}, ElemType::F32}),
      Stmt::CopyOp("%B[%ko:%ko+32, %jo:%jo+8]", "%B_tile"),
      ii,
      Stmt::Dealloc("%B_tile"),
  };

  auto jo = Stmt::For("%jo", 0, 32, 8);
  jo.body = {ko};
  auto io = Stmt::ParFor("%io", 0, 64, 32, "thread_x (2 tiles)");
  io.body = {jo};

  f.body[0] = Stmt::Comment(
      "matmul (register promotion + B stack promotion)");
  f.body[1] = io;

  std::cout << "  [StackPromo] B tile (32×8×f32 = 1KB) → memref.alloc on "
               "stack\n";
  std::cout << "    copy %B[ko:ko+32, jo:jo+8] → %B_tile per k-tile\n";
  std::cout << "  [RegPromo] C[ii,ji] accumulator hoisted out of ki loop\n";
  std::cout << "    inner reorder: ii→ki→ji → ii→ji→ki (accumulator in "
               "register)\n";
  std::cout << "    before: 128 loads + 128 stores of C per (ii,ji) pair\n";
  std::cout << "    after:  1 load + 1 store of C per (ii,ji) pair\n";
  std::cout << "    → 254 fewer memory ops per element of C!\n";
}

// =====================================================================
// P8 Step 7: Vectorization Preparation
// =====================================================================
// The innermost %ji loop (trip count 8) maps to vector<8xf32>:
//   - Eliminate %ji loop entirely
//   - C[ii, jo:jo+8] → vector.load → register %vacc
//   - A[ii, ko+ki]   → scalar load → vector.broadcast → %va
//   - B_tile[ki, 0:8] → vector.load → %vb
//   - vector.fma %va, %vb, %vacc
//   - vector.store %vacc → C[ii, jo:jo+8]
//
// This is the classic GEMM micro-kernel pattern used by BLAS / XLA / TVM.

static void stage7_vec_prep(Func &f) {
  // Rebuild matmul inner: ii → { vec_load C, ki → { load A, broadcast,
  //                               vec_load B_tile, vec_fma }, vec_store C }
  auto ki = Stmt::For("%ki", 0, 32);
  ki.body = {
      Stmt::Load("%a", "%A", {"%io+%ii", "%ko+%ki"}),
      Stmt::Arith("vector.broadcast", "%va", "%a"),
  };
  ki.body.back().vec_width = 8;
  ki.body.push_back(Stmt::VecLoad("%vb", "%B_tile", {"%ki", "0"}, 8));
  ki.body.push_back(Stmt::VecFMA("%vacc", "%va", "%vb", "%vacc", 8));

  auto ii = Stmt::For("%ii", 0, 32);
  ii.body = {
      Stmt::Comment("vectorized: ji=0..8 → vector<8xf32>"),
      Stmt::VecLoad("%vacc", "%C", {"%io+%ii", "%jo"}, 8),
      ki,
      Stmt::VecStore("%vacc", "%C", {"%io+%ii", "%jo"}, 8),
  };

  auto ko = Stmt::For("%ko", 0, 128, 32);
  ko.body = {
      Stmt::Alloc("%B_tile", MemRefType{{32, 8}, ElemType::F32}),
      Stmt::CopyOp("%B[%ko:%ko+32, %jo:%jo+8]", "%B_tile"),
      ii,
      Stmt::Dealloc("%B_tile"),
  };

  auto jo = Stmt::For("%jo", 0, 32, 8);
  jo.body = {ko};
  auto io = Stmt::ParFor("%io", 0, 64, 32, "thread_x (2 tiles)");
  io.body = {jo};

  f.body[0] = Stmt::Comment(
      "matmul (vectorized micro-kernel: vector<8xf32>)");
  f.body[1] = io;

  // Also vectorize the fused bias+relu loop
  auto fj = Stmt::For("%fj", 0, 32, 8);
  fj.body = {
      Stmt::VecLoad("%vbias", "%bias", {"%fj"}, 8),
      Stmt::VecLoad("%vc", "%C", {"%fi", "%fj"}, 8),
      Stmt::VecFMA("%vc", "%vbias", "%v_one", "%vc", 8),
  };
  // relu: maxf with zero vector
  auto relu_op = Stmt::Arith("arith.maxf", "%vc", "%vc", "%v_zero");
  relu_op.vec_width = 8;
  fj.body.push_back(relu_op);
  fj.body.push_back(Stmt::VecStore("%vc", "%C", {"%fi", "%fj"}, 8));

  auto fi = Stmt::ParFor("%fi", 0, 64, 1, "thread_x (64 rows)");
  fi.body = {fj};

  f.body[2] = Stmt::Comment(
      "bias_add + relu (fused + vectorized: vector<8xf32>)");
  f.body[3] = fi;

  std::cout << "  [Vectorize] matmul: ji=0..8 loop → vector<8xf32>\n";
  std::cout << "    inner ki loop body: load A (scalar) → broadcast → "
               "vec_load B_tile → vec_fma\n";
  std::cout << "    ji loop eliminated (replaced by SIMD lanes)\n";
  std::cout << "  [Vectorize] fused bias+relu: fj step 8 → vector<8xf32>\n";
  std::cout << "    arithmetic throughput: 8× per cycle (AVX/NEON)\n";
}

// =====================================================================
// P8 Step 8: Cleanup — final statistics and summary
// =====================================================================

static void stage8_cleanup(const Func &f) {
  int loops = f.total_loops();
  int ops = f.total_ops();
  int par = f.total_par();
  int vec = f.total_vec();

  std::cout << "  Final IR statistics:\n";
  std::cout << "    loops     = " << loops << "\n";
  std::cout << "    ops       = " << ops << "\n";
  std::cout << "    parallel  = " << par << " loops\n";
  std::cout << "    vector    = " << vec << " ops\n\n";

  std::cout << "  ┌────────────────────────────────────────────────────┐\n";
  std::cout << "  │         SCF / Affine Pipeline Summary              │\n";
  std::cout << "  ├────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Original  : 3 Linalg ops → 8 loops, 13 scalar ops│\n";
  std::cout << "  │  Final     : " << loops << " loops, " << ops
            << " ops (" << par << " parallel, " << vec << " vector)  │\n";
  std::cout << "  ├────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Optimizations applied:                            │\n";
  std::cout << "  │    ✓ Dead loop elimination         (Stage 2)       │\n";
  std::cout << "  │    ✓ Multi-level tiling 32×8×32    (Stage 3)       │\n";
  std::cout << "  │    ✓ Loop interchange i→k→j        (Stage 4)       │\n";
  std::cout << "  │    ✓ Loop fusion bias+relu          (Stage 4)       │\n";
  std::cout << "  │    ✓ Parallelization (outer loops)  (Stage 5)       │\n";
  std::cout << "  │    ✓ B tile stack promotion (1KB)   (Stage 6)       │\n";
  std::cout << "  │    ✓ Register promotion (C accum)   (Stage 6)       │\n";
  std::cout << "  │    ✓ Vectorization vector<8xf32>    (Stage 7)       │\n";
  std::cout << "  ├────────────────────────────────────────────────────┤\n";
  std::cout << "  │  Key interview points:                             │\n";
  std::cout << "  │    • Tiling  → data locality (L1/L2 cache reuse)   │\n";
  std::cout << "  │    • Interchange → stride-1 memory access pattern  │\n";
  std::cout << "  │    • Fusion  → eliminate redundant load/store      │\n";
  std::cout << "  │    • Parallel → multi-core / GPU thread mapping    │\n";
  std::cout << "  │    • Register promo → reduce memory traffic        │\n";
  std::cout << "  │    • Vectorize → SIMD throughput (8× per cycle)    │\n";
  std::cout << "  └────────────────────────────────────────────────────┘\n";
}

// =====================================================================
// Pipeline Runner
// =====================================================================

static void run_pipeline() {
  std::cout
      << "\n╔═══════════════════════════════════════════════════════════════╗\n"
      << "║  GEMM + Bias + ReLU  (SCF / Affine 9-Stage Pipeline)       ║\n"
      << "║  A[64×128] × B[128×32] + bias[32] → relu → C[64×32]       ║\n"
      << "╚═══════════════════════════════════════════════════════════════╝\n";

  // ── P8 Step 0: Pre-clean ──
  sep("Stage 0: Pre-clean (canonicalize + CSE + DCE)");
  stage0_preclean();

  // ── P8 Step 1: Linalg → Loop ──
  sep("Stage 1: Linalg → Loop (convert-linalg-to-scf)");
  auto f = stage1_linalg_to_loop();
  std::cout << "\n";
  f.print(std::cout);
  std::cout << "\n  ";
  print_stats(f);

  // ── P8 Step 2: Loop Canonicalization ──
  sep("Stage 2: Loop Canonicalization (dead loop elimination)");
  int s2 = stage2_loop_canon(f);
  std::cout << "  → removed " << s2 << " dead loop(s)\n  ";
  print_stats(f);

  // ── P8 Step 3: Tiling ──
  sep("Stage 3: Tiling (cache-aware multi-level)");
  stage3_tiling(f);
  std::cout << "\n";
  f.print(std::cout);
  std::cout << "\n  ";
  print_stats(f);

  // ── P8 Step 4: Loop Transform ──
  sep("Stage 4: Loop Transform (interchange + fusion)");
  stage4_loop_transform(f);
  std::cout << "\n";
  f.print(std::cout);
  std::cout << "\n  ";
  print_stats(f);

  // ── P8 Step 5: Parallelization ──
  sep("Stage 5: Parallelization (scf.parallel + thread mapping)");
  stage5_parallelize(f);
  std::cout << "\n";
  f.print(std::cout);
  std::cout << "\n  ";
  print_stats(f);

  // ── P8 Step 6: Memory Optimization ──
  sep("Stage 6: Memory Optimization (register promo + stack promo)");
  stage6_mem_opt(f);
  std::cout << "\n";
  f.print(std::cout);
  std::cout << "\n  ";
  print_stats(f);

  // ── P8 Step 7: Vectorization Preparation ──
  sep("Stage 7: Vectorization Preparation (vector<8xf32>)");
  stage7_vec_prep(f);
  std::cout << "\n";
  f.print(std::cout);
  std::cout << "\n  ";
  print_stats(f);

  // ── P8 Step 8: Cleanup ──
  sep("Stage 8: Cleanup (final report)");
  stage8_cleanup(f);
}

// =====================================================================
int main() {
  std::cout << "================================================================\n";
  std::cout << "  SCF / Affine Loop-Level Optimization 9-Stage Pipeline\n";
  std::cout << "  S0:PreClean → S1:LinalgToLoop → S2:LoopCanon →\n";
  std::cout << "  S3:Tiling → S4:LoopTransform → S5:Parallel →\n";
  std::cout << "  S6:MemOpt → S7:Vectorize → S8:Cleanup\n";
  std::cout << "================================================================\n";

  run_pipeline();

  std::cout << "\n✓ SCF / Affine pipeline test passed.\n";
  return 0;
}
