// run_gpu_codegen.cpp — GPU Code Generation 7-Stage Pipeline
//
// Lowers parallel computation to GPU kernel code:
//
//   Stage 0: Parallel Detection   — identify parallelizable ops (linalg.generic, scf.parallel)
//   Stage 1: Thread Mapping       — map parallel dims → GPU grid/block (blockIdx/threadIdx)
//   Stage 2: GPU Dialect Emission — emit gpu.launch_func, gpu.block_id, gpu.thread_id
//   Stage 3: Shared Memory Tiling — insert shared memory allocation + tiled loads + barriers
//   Stage 4: NVVM Lowering        — gpu dialect → NVVM (tid/ctaid/barrier0, LLVM pointer ops)
//   Stage 5: PTX Emission         — NVVM → PTX assembly text
//   Stage 6: Occupancy Analysis   — estimate register/smem usage, compute SM occupancy
//
// Test case: GEMM C[M,N] = A[M,K] · B[K,N]  with shared-memory tiling.
//
// Pure C++17, header-only IR, no external dependencies.

#include "gpu_codegen_ir.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace gpu_ir;

static void sep(const char *title) {
  std::cout << "\n  ──── " << title << " ────\n\n";
}

// =====================================================================
// Problem setup: GEMM  C[M,N] += A[M,K] * B[K,N]
// =====================================================================

static constexpr int M = 1024, N = 1024, K = 512;
static constexpr int TILE_M = 64, TILE_N = 64, TILE_K = 32;
static constexpr int BLOCK_X = 16, BLOCK_Y = 16;
static constexpr int THREAD_TILE_M = TILE_M / BLOCK_Y;
static constexpr int THREAD_TILE_N = TILE_N / BLOCK_X;

// =====================================================================
// Stage 0: Parallel Detection
// =====================================================================
// Analyze linalg.generic / scf.parallel to identify parallel iteration dims.
// For GEMM:  iterator_types = [parallel, parallel, reduction]
//            indexing_maps  = [(m,k)->A, (k,n)->B, (m,n)->C]

static void stage0_parallel_detection() {
  sep("Stage 0: Parallel Detection");

  std::cout << "  Input: linalg.generic GEMM operation\n\n";
  std::cout << "  linalg.generic {\n";
  std::cout << "    indexing_maps = [\n";
  std::cout << "      affine_map<(m, n, k) -> (m, k)>,   // A\n";
  std::cout << "      affine_map<(m, n, k) -> (k, n)>,   // B\n";
  std::cout << "      affine_map<(m, n, k) -> (m, n)>    // C\n";
  std::cout << "    ],\n";
  std::cout << "    iterator_types = [\"parallel\", \"parallel\", \"reduction\"]\n";
  std::cout << "  } ins(%A, %B) outs(%C) {\n";
  std::cout << "    ^bb0(%a: f32, %b: f32, %c: f32):\n";
  std::cout << "      %p = arith.mulf %a, %b : f32\n";
  std::cout << "      %s = arith.addf %c, %p : f32\n";
  std::cout << "      linalg.yield %s : f32\n";
  std::cout << "  }\n\n";

  std::cout << "  Analysis result:\n";
  std::cout << "    dim 'm' (size " << M << "): PARALLEL  → mappable to GPU\n";
  std::cout << "    dim 'n' (size " << N << "): PARALLEL  → mappable to GPU\n";
  std::cout << "    dim 'k' (size " << K << "): REDUCTION → sequential (inner loop)\n\n";

  std::cout << "  Decision: map (m, n) → GPU grid+block, k → sequential loop inside kernel\n";
  std::cout << "  Parallelism: " << M << " × " << N << " = "
            << (M * N) << " independent output elements\n";
}

// =====================================================================
// Stage 1: Thread Mapping
// =====================================================================
// Map parallel dimensions to GPU hierarchy:
//   m → gridDim.y * blockDim.y  (outer tiles → blocks, inner → threads)
//   n → gridDim.x * blockDim.x

static TileStrategy stage1_thread_mapping() {
  sep("Stage 1: Thread Mapping");

  TileStrategy ts;
  ts.op_name = "gemm";
  ts.tile_sizes = {TILE_M, TILE_N, TILE_K};
  ts.block_x = BLOCK_X;
  ts.block_y = BLOCK_Y;
  ts.use_shared_mem = true;
  ts.shared_tile_m = TILE_M;
  ts.shared_tile_n = TILE_N;
  ts.shared_tile_k = TILE_K;

  KernelConfig cfg;
  cfg.name = "gemm_kernel";
  cfg.grid = {N / TILE_N, M / TILE_M, 1};
  cfg.block = {BLOCK_X, BLOCK_Y, 1};
  cfg.shared_mem_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * 4;
  cfg.reg_per_thread = THREAD_TILE_M * THREAD_TILE_N + 4;

  std::cout << "  Tiling strategy:\n";
  ts.print(std::cout);
  std::cout << "\n  Thread-level work: each thread computes "
            << THREAD_TILE_M << "×" << THREAD_TILE_N
            << " output elements\n\n";

  std::cout << "  Kernel launch configuration:\n";
  cfg.print(std::cout);
  std::cout << "\n";

  std::cout << "  GPU hierarchy mapping:\n";
  std::cout << "    blockIdx.y  × blockDim.y → row tile [0, " << M << ") step " << TILE_M << "\n";
  std::cout << "    blockIdx.x  × blockDim.x → col tile [0, " << N << ") step " << TILE_N << "\n";
  std::cout << "    threadIdx.y → row within tile [0, " << TILE_M << ") step " << THREAD_TILE_M << "\n";
  std::cout << "    threadIdx.x → col within tile [0, " << TILE_N << ") step " << THREAD_TILE_N << "\n";
  std::cout << "    sequential  → k reduction  [0, " << K << ") step " << TILE_K << "\n";

  return ts;
}

// =====================================================================
// Stage 2: GPU Dialect Emission
// =====================================================================
// Emit gpu.launch_func with block/thread id ops.

static void stage2_gpu_dialect(const TileStrategy &ts) {
  sep("Stage 2: GPU Dialect Emission");

  int grid_x = N / TILE_N, grid_y = M / TILE_M;

  std::cout << "  // Host side: launch kernel\n";
  std::cout << "  gpu.launch_func @gemm_module::@gemm_kernel\n";
  std::cout << "      blocks in (" << grid_x << ", " << grid_y << ", 1)\n";
  std::cout << "      threads in (" << BLOCK_X << ", " << BLOCK_Y << ", 1)\n";
  std::cout << "      dynamic_shared_memory_size "
            << (TILE_M * TILE_K + TILE_K * TILE_N) * 4 << "\n";
  std::cout << "      args(%A, %B, %C : memref<" << M << "x" << K << "xf32>,\n";
  std::cout << "                         memref<" << K << "x" << N << "xf32>,\n";
  std::cout << "                         memref<" << M << "x" << N << "xf32>)\n\n";

  std::cout << "  // Device kernel in GPU dialect:\n";
  std::cout << "  gpu.module @gemm_module {\n";
  std::cout << "    gpu.func @gemm_kernel(%A: memref<" << M << "x" << K << "xf32>,\n";
  std::cout << "                          %B: memref<" << K << "x" << N << "xf32>,\n";
  std::cout << "                          %C: memref<" << M << "x" << N << "xf32>)\n";
  std::cout << "        workgroup(%sA: memref<" << TILE_M << "x" << TILE_K
            << "xf32, #gpu.address_space<workgroup>>,\n";
  std::cout << "                  %sB: memref<" << TILE_K << "x" << TILE_N
            << "xf32, #gpu.address_space<workgroup>>)\n";
  std::cout << "    {\n";
  std::cout << "      %bx = gpu.block_id x          // blockIdx.x\n";
  std::cout << "      %by = gpu.block_id y          // blockIdx.y\n";
  std::cout << "      %tx = gpu.thread_id x         // threadIdx.x\n";
  std::cout << "      %ty = gpu.thread_id y         // threadIdx.y\n\n";

  std::cout << "      // Global row/col base for this block\n";
  std::cout << "      %row_base = arith.muli %by, " << TILE_M << "\n";
  std::cout << "      %col_base = arith.muli %bx, " << TILE_N << "\n\n";

  std::cout << "      // Per-thread accumulator registers (" << THREAD_TILE_M << "x" << THREAD_TILE_N << " f32)\n";
  for (int i = 0; i < THREAD_TILE_M; ++i)
    for (int j = 0; j < THREAD_TILE_N; ++j)
      std::cout << "      %acc_" << i << "_" << j << " = arith.constant 0.0 : f32\n";

  std::cout << "\n      // K-tile loop\n";
  std::cout << "      scf.for %ko = 0 to " << K << " step " << TILE_K << " {\n\n";

  std::cout << "        // === Cooperative load: global → shared ===\n";
  std::cout << "        // Each thread loads " << (TILE_M * TILE_K) / (BLOCK_X * BLOCK_Y)
            << " elements of A_tile, "
            << (TILE_K * TILE_N) / (BLOCK_X * BLOCK_Y) << " elements of B_tile\n";
  std::cout << "        %lin_id = arith.addi (arith.muli %ty, " << BLOCK_X << "), %tx\n";
  std::cout << "        // ... (vectorized cooperative loads) ...\n";
  std::cout << "        memref.store %a_val, %sA[%local_row, %local_col]\n";
  std::cout << "        memref.store %b_val, %sB[%local_row, %local_col]\n\n";

  std::cout << "        gpu.barrier  // __syncthreads()\n\n";

  std::cout << "        // === Compute: shared memory → registers ===\n";
  std::cout << "        scf.for %ki = 0 to " << TILE_K << " step 1 {\n";
  for (int i = 0; i < std::min(THREAD_TILE_M, 2); ++i) {
    std::cout << "          %a_" << i << " = memref.load %sA[%ty*" << THREAD_TILE_M
              << "+" << i << ", %ki]\n";
  }
  std::cout << "          // ... (load remaining A fragments)\n";
  for (int j = 0; j < std::min(THREAD_TILE_N, 2); ++j) {
    std::cout << "          %b_" << j << " = memref.load %sB[%ki, %tx*" << THREAD_TILE_N
              << "+" << j << "]\n";
  }
  std::cout << "          // ... (load remaining B fragments)\n";
  std::cout << "          // Outer product: acc[i][j] += a[i] * b[j]\n";
  std::cout << "          %acc_0_0 = arith.addf %acc_0_0, (arith.mulf %a_0, %b_0)\n";
  std::cout << "          // ... (" << THREAD_TILE_M * THREAD_TILE_N << " FMAs total)\n";
  std::cout << "        }\n\n";

  std::cout << "        gpu.barrier  // sync before next tile load\n";
  std::cout << "      }\n\n";

  std::cout << "      // === Store: registers → global ===\n";
  std::cout << "      // Write back " << THREAD_TILE_M << "x" << THREAD_TILE_N << " accumulators to C\n";
  std::cout << "      memref.store %acc_0_0, %C[%row_base + %ty*" << THREAD_TILE_M
            << "+0, %col_base + %tx*" << THREAD_TILE_N << "+0]\n";
  std::cout << "      // ... (remaining stores)\n";
  std::cout << "      gpu.return\n";
  std::cout << "    }\n";
  std::cout << "  }\n";
}

// =====================================================================
// Stage 3: Shared Memory Tiling Analysis
// =====================================================================
// Detailed analysis of data movement and bank conflict avoidance.

static void stage3_shared_memory() {
  sep("Stage 3: Shared Memory Tiling Analysis");

  int smem_a = TILE_M * TILE_K * 4;
  int smem_b = TILE_K * TILE_N * 4;
  int total_smem = smem_a + smem_b;

  std::cout << "  Shared memory layout:\n";
  std::cout << "    sA: " << TILE_M << " × " << TILE_K << " × 4B = "
            << smem_a << " bytes (" << smem_a / 1024 << " KB)\n";
  std::cout << "    sB: " << TILE_K << " × " << TILE_N << " × 4B = "
            << smem_b << " bytes (" << smem_b / 1024 << " KB)\n";
  std::cout << "    Total: " << total_smem << " bytes (" << total_smem / 1024 << " KB)\n\n";

  std::cout << "  Bank conflict analysis (32 banks, 4-byte granularity):\n";
  std::cout << "    sA access: threads in warp read sA[ty*" << THREAD_TILE_M << "+i, ki]\n";
  std::cout << "      → stride = " << TILE_K << " × 4 = " << TILE_K * 4 << " bytes\n";
  std::cout << "      → bank = (addr / 4) % 32\n";
  if (TILE_K % 32 == 0) {
    std::cout << "      ⚠ TILE_K=" << TILE_K << " is multiple of 32 → bank conflicts!\n";
    std::cout << "      → Fix: pad sA to " << TILE_M << "×" << (TILE_K + 1) << " (add 1 column)\n";
  } else {
    std::cout << "      ✓ No bank conflicts (stride not multiple of 32)\n";
  }
  std::cout << "    sB access: threads in warp read sB[ki, tx*" << THREAD_TILE_N << "+j]\n";
  std::cout << "      → consecutive threads access consecutive columns → no conflict\n\n";

  std::cout << "  Cooperative load pattern (coalescing analysis):\n";
  int elems_per_thread_a = (TILE_M * TILE_K) / (BLOCK_X * BLOCK_Y);
  int elems_per_thread_b = (TILE_K * TILE_N) / (BLOCK_X * BLOCK_Y);
  std::cout << "    Threads per block: " << BLOCK_X * BLOCK_Y << "\n";
  std::cout << "    A elements/thread: " << elems_per_thread_a << "\n";
  std::cout << "    B elements/thread: " << elems_per_thread_b << "\n";
  std::cout << "    ✓ Row-major cooperative load → coalesced 128-byte transactions\n\n";

  std::cout << "  Double buffering opportunity:\n";
  std::cout << "    Buffer 0: load tile[ko+1] while computing tile[ko]\n";
  std::cout << "    Buffer 1: swap roles each iteration\n";
  std::cout << "    Benefit: overlap global memory latency (~400 cycles) with compute\n";
  std::cout << "    Cost: 2× shared memory (" << total_smem * 2 / 1024 << " KB total)\n";
}

// =====================================================================
// Stage 4: NVVM Lowering
// =====================================================================
// GPU dialect → NVVM dialect (LLVM dialect + nvvm intrinsics)

static void stage4_nvvm_lowering() {
  sep("Stage 4: NVVM Lowering");

  std::cout << "  Lowering rules (gpu → nvvm):\n\n";

  struct Rule {
    const char *from;
    const char *to;
    const char *note;
  };
  Rule rules[] = {
    {"gpu.block_id x",     "nvvm.read.ptx.sreg.ctaid.x",  "%ctaid.x"},
    {"gpu.block_id y",     "nvvm.read.ptx.sreg.ctaid.y",  "%ctaid.y"},
    {"gpu.thread_id x",    "nvvm.read.ptx.sreg.tid.x",    "%tid.x"},
    {"gpu.thread_id y",    "nvvm.read.ptx.sreg.tid.y",    "%tid.y"},
    {"gpu.block_dim x",    "nvvm.read.ptx.sreg.ntid.x",   "blockDim.x"},
    {"gpu.barrier",        "nvvm.barrier0",                "__syncthreads()"},
    {"memref.load (global)","llvm.load ptr addrspace(1)",  "global memory"},
    {"memref.store(global)","llvm.store ptr addrspace(1)", "global memory"},
    {"memref.load (shared)","llvm.load ptr addrspace(3)",  "shared memory"},
    {"memref.store(shared)","llvm.store ptr addrspace(3)", "shared memory"},
    {"memref.alloc shared", "llvm.mlir.global @smem(dense<0.0>)", "module-level"},
    {"arith.addf",         "llvm.fadd float",              ""},
    {"arith.mulf",         "llvm.fmul float",              ""},
    {"arith.muli",         "llvm.mul i32",                 ""},
    {"arith.index_cast",   "llvm.trunc/zext i32",          "index → i32"},
    {"gpu.func",           "llvm.func ... attributes {nvvm.kernel}", ""},
  };

  std::cout << "  " << std::left << std::setw(26) << "GPU Dialect"
            << std::setw(38) << "NVVM / LLVM Dialect"
            << "Note\n";
  std::cout << "  " << std::string(80, '-') << "\n";
  for (auto &r : rules) {
    std::cout << "  " << std::left << std::setw(26) << r.from
              << std::setw(38) << r.to << r.note << "\n";
  }

  std::cout << "\n  NVVM kernel IR (excerpt):\n\n";
  std::cout << "  llvm.func @gemm_kernel(%A: !llvm.ptr {llvm.addrspace = 1},\n";
  std::cout << "                         %B: !llvm.ptr {llvm.addrspace = 1},\n";
  std::cout << "                         %C: !llvm.ptr {llvm.addrspace = 1})\n";
  std::cout << "      attributes {nvvm.kernel, nvvm.maxntid = [" << BLOCK_X << ", " << BLOCK_Y << ", 1]} {\n";
  std::cout << "    %tid_x  = nvvm.read.ptx.sreg.tid.x   : i32\n";
  std::cout << "    %tid_y  = nvvm.read.ptx.sreg.tid.y   : i32\n";
  std::cout << "    %ctaid_x = nvvm.read.ptx.sreg.ctaid.x : i32\n";
  std::cout << "    %ctaid_y = nvvm.read.ptx.sreg.ctaid.y : i32\n\n";
  std::cout << "    %row_base = llvm.mul %ctaid_y, " << TILE_M << " : i32\n";
  std::cout << "    %col_base = llvm.mul %ctaid_x, " << TILE_N << " : i32\n\n";
  std::cout << "    // Shared memory as module-level global\n";
  std::cout << "    %sA_ptr = llvm.mlir.addressof @shared_A : !llvm.ptr<3>\n";
  std::cout << "    %sB_ptr = llvm.mlir.addressof @shared_B : !llvm.ptr<3>\n\n";
  std::cout << "    // ... (k-tile loop with barrier0, same structure as Stage 2)\n";
  std::cout << "    nvvm.barrier0   // replaces gpu.barrier\n";
  std::cout << "    // ... (compute + store back)\n";
  std::cout << "    llvm.return\n";
  std::cout << "  }\n";
}

// =====================================================================
// Stage 5: PTX Emission
// =====================================================================

static void stage5_ptx_emission() {
  sep("Stage 5: PTX Emission");

  std::cout << "  NVVM IR → PTX (via LLVM NVPTX backend):\n\n";

  std::vector<PTXInstr> ptx;

  ptx.push_back({".version 7.8", {}, ""});
  ptx.push_back({".target sm_80", {}, "Ampere architecture"});
  ptx.push_back({".address_size 64", {}, ""});
  ptx.push_back({"", {}, ""});
  ptx.push_back({".shared .align 4 .b8", {"shared_A[" + std::to_string(TILE_M * TILE_K * 4) + "]"}, "sA tile"});
  ptx.push_back({".shared .align 4 .b8", {"shared_B[" + std::to_string(TILE_K * TILE_N * 4) + "]"}, "sB tile"});
  ptx.push_back({"", {}, ""});
  ptx.push_back({".visible .entry gemm_kernel(", {}, ""});
  ptx.push_back({"  .param .u64 A_ptr,", {}, ""});
  ptx.push_back({"  .param .u64 B_ptr,", {}, ""});
  ptx.push_back({"  .param .u64 C_ptr", {}, ""});
  ptx.push_back({")", {}, ""});
  ptx.push_back({"{", {}, ""});
  ptx.push_back({"  .reg .f32", {"<" + std::to_string(THREAD_TILE_M * THREAD_TILE_N + 8) + ">"}, "acc + temp"});
  ptx.push_back({"  .reg .u32", {"<16>"}, "index registers"});
  ptx.push_back({"  .reg .u64", {"<8>"}, "address registers"});
  ptx.push_back({"  .reg .pred", {"<4>"}, "predicate registers"});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"  mov.u32", {"r0", "%tid.x"}, "threadIdx.x"});
  ptx.push_back({"  mov.u32", {"r1", "%tid.y"}, "threadIdx.y"});
  ptx.push_back({"  mov.u32", {"r2", "%ctaid.x"}, "blockIdx.x"});
  ptx.push_back({"  mov.u32", {"r3", "%ctaid.y"}, "blockIdx.y"});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"  // Row/col base", {}, ""});
  ptx.push_back({"  mad.lo.u32", {"r4", "r3", std::to_string(TILE_M), "0"}, "row_base"});
  ptx.push_back({"  mad.lo.u32", {"r5", "r2", std::to_string(TILE_N), "0"}, "col_base"});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"  // Zero accumulators", {}, ""});
  for (int i = 0; i < std::min(THREAD_TILE_M * THREAD_TILE_N, 4); ++i)
    ptx.push_back({"  mov.f32", {"f" + std::to_string(i), "0f00000000"}, "acc[" + std::to_string(i) + "]"});
  ptx.push_back({"  // ... (remaining accumulators)", {}, ""});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"K_LOOP:", {}, "k-tile loop"});
  ptx.push_back({"  // Cooperative load A_tile: global → shared", {}, ""});
  ptx.push_back({"  ld.global.v4.f32", {"{f20,f21,f22,f23}", "[rd0]"}, "128-bit coalesced"});
  ptx.push_back({"  st.shared.v4.f32", {"[shared_A + offset]", "{f20,f21,f22,f23}"}, ""});
  ptx.push_back({"  // ... (cooperative loads for B_tile)", {}, ""});
  ptx.push_back({"  bar.sync 0", {}, "__syncthreads()"});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"  // Inner k loop: compute from shared", {}, ""});
  ptx.push_back({"INNER_K:", {}, ""});
  ptx.push_back({"  ld.shared.f32", {"f24", "[sA_addr]"}, "a[i]"});
  ptx.push_back({"  ld.shared.f32", {"f25", "[sB_addr]"}, "b[j]"});
  ptx.push_back({"  fma.rn.f32", {"f0", "f24", "f25", "f0"}, "acc[0] += a*b"});
  ptx.push_back({"  // ... (remaining FMAs)", {}, ""});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"  bar.sync 0", {}, "sync before next tile"});
  ptx.push_back({"  // ... (branch back to K_LOOP)", {}, ""});
  ptx.push_back({"", {}, ""});
  ptx.push_back({"  // Store results: registers → global", {}, ""});
  ptx.push_back({"  st.global.f32", {"[rd4]", "f0"}, "C[row][col]"});
  ptx.push_back({"  // ... (remaining stores)", {}, ""});
  ptx.push_back({"  ret", {}, ""});
  ptx.push_back({"}", {}, ""});

  for (auto &inst : ptx)
    std::cout << inst.str() << "\n";
}

// =====================================================================
// Stage 6: Occupancy Analysis
// =====================================================================

static void stage6_occupancy() {
  sep("Stage 6: Occupancy Analysis");

  KernelConfig cfg;
  cfg.name = "gemm_kernel";
  cfg.grid = {N / TILE_N, M / TILE_M, 1};
  cfg.block = {BLOCK_X, BLOCK_Y, 1};
  cfg.shared_mem_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * 4;
  cfg.reg_per_thread = THREAD_TILE_M * THREAD_TILE_N + 8;

  std::cout << "  Target: NVIDIA A100 (SM 8.0)\n\n";

  std::cout << "  SM Limits:\n";
  std::cout << "    Max threads/SM:  2048\n";
  std::cout << "    Max blocks/SM:   32\n";
  std::cout << "    Max registers:   65536\n";
  std::cout << "    Max shared mem:  164 KB (configurable: 48KB default)\n\n";

  std::cout << "  Kernel resource usage:\n";
  std::cout << "    Threads/block:   " << cfg.block.total() << "\n";
  std::cout << "    Registers/thread:" << cfg.reg_per_thread << "\n";
  std::cout << "    Shared mem/block:" << cfg.shared_mem_bytes << " bytes ("
            << cfg.shared_mem_bytes / 1024 << " KB)\n";
  std::cout << "    Total registers/block: "
            << cfg.block.total() * cfg.reg_per_thread << "\n\n";

  int blocks_by_threads = 2048 / cfg.block.total();
  int blocks_by_regs = 65536 / (cfg.block.total() * cfg.reg_per_thread);
  int blocks_by_smem = 49152 / cfg.shared_mem_bytes;

  std::cout << "  Occupancy limiters:\n";
  std::cout << "    By threads:  2048 / " << cfg.block.total()
            << " = " << blocks_by_threads << " blocks\n";
  std::cout << "    By registers: 65536 / " << cfg.block.total() * cfg.reg_per_thread
            << " = " << blocks_by_regs << " blocks\n";
  std::cout << "    By shared mem: 49152 / " << cfg.shared_mem_bytes
            << " = " << blocks_by_smem << " blocks\n";

  int actual = std::min({blocks_by_threads, blocks_by_regs, blocks_by_smem, 32});
  double occ = cfg.occupancy_estimate();
  std::cout << "\n  Active blocks/SM: " << actual << "\n";
  std::cout << "  Occupancy: " << static_cast<int>(occ * 100) << "%\n\n";

  std::cout << "  Bottleneck: "
            << (blocks_by_regs <= blocks_by_threads && blocks_by_regs <= blocks_by_smem
                ? "REGISTERS" : blocks_by_smem <= blocks_by_threads
                ? "SHARED MEMORY" : "THREADS")
            << "\n\n";

  std::cout << "  Optimization suggestions:\n";
  if (blocks_by_regs < blocks_by_threads)
    std::cout << "    → Reduce register usage: smaller thread tile or use --maxrregcount\n";
  if (blocks_by_smem < blocks_by_threads)
    std::cout << "    → Reduce shared memory: smaller tile_k or use multi-stage pipeline\n";
  std::cout << "    → Consider warp-level MMA (Tensor Cores) for higher throughput\n";
  std::cout << "    → Use async copy (cp.async) for global→shared to hide latency\n";

  int total_blocks = cfg.grid.total();
  int num_sms = 108;
  int waves = (total_blocks + num_sms * actual - 1) / (num_sms * actual);
  std::cout << "\n  Wave analysis (A100, " << num_sms << " SMs):\n";
  std::cout << "    Total blocks: " << total_blocks << "\n";
  std::cout << "    Blocks/SM: " << actual << "\n";
  std::cout << "    Waves: " << waves << "\n";
  if (total_blocks % (num_sms * actual) != 0) {
    int tail = total_blocks % (num_sms * actual);
    std::cout << "    ⚠ Tail effect: last wave has " << tail << "/" << num_sms * actual
              << " blocks (" << tail * 100 / (num_sms * actual) << "% utilization)\n";
  }
}

// =====================================================================
// main
// =====================================================================

int main() {
  std::cout << "========================================================\n";
  std::cout << " Stage 13: GPU Code Generation Pipeline\n";
  std::cout << " GEMM C[" << M << "," << N << "] = A[" << M << "," << K
            << "] · B[" << K << "," << N << "]\n";
  std::cout << "========================================================\n";

  stage0_parallel_detection();
  auto ts = stage1_thread_mapping();
  stage2_gpu_dialect(ts);
  stage3_shared_memory();
  stage4_nvvm_lowering();
  stage5_ptx_emission();
  stage6_occupancy();

  std::cout << "\n========================================================\n";
  std::cout << " GPU Code Generation Pipeline Complete!\n";
  std::cout << "========================================================\n";
  return 0;
}
