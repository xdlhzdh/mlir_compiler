# =============================================================================
# src/mlir/gpu — MLIR 路线 AI 编译器 Pipeline（Px 编号）
#
# 标准阶段 Px（目录名前的数字仅作排序，以 Px 为准）：
#
#   P1  1_onnx_parse/         L0：ONNX GraphProto 解析
#   P2  2_onnx_to_ir/         L1：ONNX → mini_ir
#   P3  3_graph_optimize/     L1：mini_ir 图优化
#   —   4_torch_to_stablehlo/ 非标准 Px：PyTorch/torch-mlir 导出 StableHLO（辅助 P5）
#   P4  5_onnx_to_stablehlo/  L1：ONNX → StableHLO lowering 练习（内含 tier 1/2/3 难度）
#   P5  6_stablehlo_passes/   L1：真实 MLIR StableHLO + Pass 插件（mlir-opt）
#       7_stablehlo_opt/       L1：StableHLO 图优化（shlo_graph，6-step Pass 链）
#   P6  8_linalg_opt/         L2：Linalg on tensor（linalg_ir）
#   P7  9_bufferize/          L2：OSB / linalg on memref（bufferize_ir）
#   P8  10_scf_affine/        L3：循环级（scf_affine_ir）
#       11_vector/             L3：向量化（vector_ir）
#   P9  12_llvm_lowering/     L4：Vector → LLVM 后端（llvm_lowering_ir）
#   P10 13_gpu_codegen/       L3/L4：GPU 映射与 PTX（概念）
#   P11 14_quantization/      横切：量化与混合精度
#   P12 15_memory_planning/   横切：内存规划
#
#   common/                   共享：onnx_proto、mini_ir.h、脚本
#
# 子 Pass 流水线内的步骤在源码注释中记为「Px Step N」，勿与目录 Px 混淆。
# =============================================================================

# ─── P1–P3：纯 C++ ONNX / mini_ir（仅依赖 Protobuf）────────────────────────

find_package(Protobuf QUIET)
if(Protobuf_FOUND)
  # common: ONNX protobuf 静态库 (编译一次, 三个可执行共享)
  set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
  add_library(onnx_proto STATIC ${COMMON_DIR}/onnx-ml.pb.cc)
  target_include_directories(onnx_proto PUBLIC ${COMMON_DIR} ${Protobuf_INCLUDE_DIRS})
  target_link_libraries(onnx_proto PUBLIC ${Protobuf_LIBRARIES})
  target_compile_options(onnx_proto PRIVATE -w)

  # P1: ONNX 解析
  add_executable(run_onnx_parse 1_onnx_parse/run_onnx_parse.cpp)
  target_include_directories(run_onnx_parse PRIVATE ${COMMON_DIR})
  target_link_libraries(run_onnx_parse onnx_proto)

  # P2: ONNX → IR Lowering
  add_executable(run_onnx_to_ir 2_onnx_to_ir/run_onnx_to_ir.cpp)
  target_include_directories(run_onnx_to_ir PRIVATE ${COMMON_DIR})
  target_link_libraries(run_onnx_to_ir onnx_proto)

  # P3: 图级优化
  add_executable(run_graph_rewrite 3_graph_optimize/run_graph_rewrite.cpp)
  target_include_directories(run_graph_rewrite PRIVATE ${COMMON_DIR})
  target_link_libraries(run_graph_rewrite onnx_proto)

  # 测试模型生成
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "python3")
  endif()

  set(TEST_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/models)
  add_custom_target(gen_test_models
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_MODEL_DIR}
    COMMAND ${Python3_EXECUTABLE} ${COMMON_DIR}/gen_test_models.py ${TEST_MODEL_DIR}
    WORKING_DIRECTORY ${COMMON_DIR}
    COMMENT "Generating test ONNX models for L1 graph exercises")

  # 一键运行 P1+P2+P3
  configure_file(
    "${COMMON_DIR}/run_graph.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_graph.cmake"
    @ONLY)

  add_custom_target(run_graph
    COMMAND ${CMAKE_COMMAND}
      -DRUN_ONNX_PARSE=$<TARGET_FILE:run_onnx_parse>
      -DRUN_ONNX_TO_IR=$<TARGET_FILE:run_onnx_to_ir>
      -DRUN_GRAPH_REWRITE=$<TARGET_FILE:run_graph_rewrite>
      -DMODEL_DIR=${TEST_MODEL_DIR}
      -P ${CMAKE_CURRENT_BINARY_DIR}/run_graph.cmake
    DEPENDS run_onnx_parse run_onnx_to_ir run_graph_rewrite gen_test_models
    COMMENT "Running full L1 Graph Level pipeline P1+P2+P3")

  # ─── P4：ONNX → StableHLO lowering 练习（tier 1/2/3，仅 Protobuf）──────────

  set(ONNX_TO_STABLEHLO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/5_onnx_to_stablehlo)

  # P4 tier 1: 基础 lowering (Add/MatMul/Conv/Reshape/Transpose)
  add_executable(run_lowering_l1 ${ONNX_TO_STABLEHLO_DIR}/run_level1.cpp)
  target_include_directories(run_lowering_l1 PRIVATE ${COMMON_DIR} ${ONNX_TO_STABLEHLO_DIR})
  target_link_libraries(run_lowering_l1 onnx_proto)

  # P4 tier 2: broadcast / dynamic shape / attribute mapping / error handling
  add_executable(run_lowering_l2 ${ONNX_TO_STABLEHLO_DIR}/run_level2.cpp)
  target_include_directories(run_lowering_l2 PRIVATE ${COMMON_DIR} ${ONNX_TO_STABLEHLO_DIR})
  target_link_libraries(run_lowering_l2 onnx_proto)

  # P4 tier 3: conversion framework / legalization pipeline / pattern-based
  add_executable(run_lowering_l3 ${ONNX_TO_STABLEHLO_DIR}/run_level3.cpp)
  target_include_directories(run_lowering_l3 PRIVATE ${COMMON_DIR} ${ONNX_TO_STABLEHLO_DIR})
  target_link_libraries(run_lowering_l3 onnx_proto)

  # Lowering 测试模型生成
  set(LOWERING_MODEL_DIR ${CMAKE_CURRENT_BINARY_DIR}/lowering_models)
  add_custom_target(gen_lowering_models
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LOWERING_MODEL_DIR}
    COMMAND ${Python3_EXECUTABLE} ${ONNX_TO_STABLEHLO_DIR}/gen_lowering_models.py ${LOWERING_MODEL_DIR}
    WORKING_DIRECTORY ${ONNX_TO_STABLEHLO_DIR}
    COMMENT "Generating lowering test ONNX models for P4 (tiers 1-3)")

  # 一键运行 P4 tier 1+2+3
  configure_file(
    "${ONNX_TO_STABLEHLO_DIR}/run_lowering.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_lowering.cmake"
    @ONLY)

  add_custom_target(run_lowering
    COMMAND ${CMAKE_COMMAND}
      -DRUN_L1=$<TARGET_FILE:run_lowering_l1>
      -DRUN_L2=$<TARGET_FILE:run_lowering_l2>
      -DRUN_L3=$<TARGET_FILE:run_lowering_l3>
      -DMODEL_DIR=${LOWERING_MODEL_DIR}
      -P ${CMAKE_CURRENT_BINARY_DIR}/run_lowering.cmake
    DEPENDS run_lowering_l1 run_lowering_l2 run_lowering_l3 gen_lowering_models
    COMMENT "Running full P4 ONNX→StableHLO lowering (tiers 1+2+3)")
else()
  message(WARNING "Protobuf not found. P1~P3 targets will not be built.")
endif()

# ─── P5（7_stablehlo_opt）：StableHLO 图优化，shlo_graph，纯 C++ ───────────

set(SHLO_OPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/7_stablehlo_opt)
add_executable(run_stablehlo_opt ${SHLO_OPT_DIR}/run_stablehlo_opt.cpp)
target_include_directories(run_stablehlo_opt PRIVATE ${SHLO_OPT_DIR})
target_compile_features(run_stablehlo_opt PRIVATE cxx_std_17)

add_custom_target(run_shlo_opt
  COMMAND $<TARGET_FILE:run_stablehlo_opt>
  DEPENDS run_stablehlo_opt
  COMMENT "Running P5 shlo_graph optimization (6-step pipeline)")

# ─── P6（8_linalg_opt）：Linalg on tensor ───────────────────────────────────

set(LINALG_OPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/8_linalg_opt)
add_executable(run_linalg_opt ${LINALG_OPT_DIR}/run_linalg_opt.cpp)
target_include_directories(run_linalg_opt PRIVATE ${LINALG_OPT_DIR})
target_compile_features(run_linalg_opt PRIVATE cxx_std_17)

add_custom_target(run_linalg
  COMMAND $<TARGET_FILE:run_linalg_opt>
  DEPENDS run_linalg_opt
  COMMENT "Running Linalg tensor-level optimization pipeline (Cluster→Tile→Fuse→DCE)")

# ─── P7（9_bufferize）：One-Shot Bufferization ───────────────────────────────

set(BUFFERIZE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/9_bufferize)
add_executable(run_bufferize ${BUFFERIZE_DIR}/run_bufferize.cpp)
target_include_directories(run_bufferize PRIVATE ${BUFFERIZE_DIR})
target_compile_features(run_bufferize PRIVATE cxx_std_17)

add_custom_target(run_buf
  COMMAND $<TARGET_FILE:run_bufferize>
  DEPENDS run_bufferize
  COMMENT "Running One-Shot Bufferization pipeline (Alias→InPlace→Rewrite→Copy→Dealloc)")

# ─── P8（10_scf_affine）：SCF/Affine 循环优化 ─────────────────────────────

set(SCF_AFFINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/10_scf_affine)
add_executable(run_scf_affine ${SCF_AFFINE_DIR}/run_scf_affine.cpp)
target_include_directories(run_scf_affine PRIVATE ${SCF_AFFINE_DIR})
target_compile_features(run_scf_affine PRIVATE cxx_std_17)

add_custom_target(run_scf
  COMMAND $<TARGET_FILE:run_scf_affine>
  DEPENDS run_scf_affine
  COMMENT "Running SCF/Affine loop-level optimization pipeline (Tile→Interchange→Fuse→Parallel→Vec)")

# ─── P8（11_vector）：Vector 向量化 ─────────────────────────────────────────

set(VECTOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/11_vector)
add_executable(run_vector_pipeline ${VECTOR_DIR}/run_vector.cpp)
target_include_directories(run_vector_pipeline PRIVATE ${VECTOR_DIR})
target_compile_features(run_vector_pipeline PRIVATE cxx_std_17)

add_custom_target(run_vec
  COMMAND $<TARGET_FILE:run_vector_pipeline>
  DEPENDS run_vector_pipeline
  COMMENT "Running Vector dialect pipeline (Precond→Core→Shaping→Opt→LLVM)")

# ─── P9（12_llvm_lowering）：Vector → LLVM 后端 ─────────────────────────────

set(LLVM_LOWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/12_llvm_lowering)
add_executable(run_llvm_lowering ${LLVM_LOWER_DIR}/run_llvm_lowering.cpp)
target_include_directories(run_llvm_lowering PRIVATE ${LLVM_LOWER_DIR})
target_compile_features(run_llvm_lowering PRIVATE cxx_std_17)

add_custom_target(run_llvm_lower
  COMMAND $<TARGET_FILE:run_llvm_lowering>
  DEPENDS run_llvm_lowering
  COMMENT "Running Vector→LLVM lowering backend pipeline (VecLower→LLVMIR→ISel→RegAlloc→Sched→Emit)")

# ─── P10（13_gpu_codegen）：GPU 代码生成（概念）──────────────────────────────

set(GPU_CODEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/13_gpu_codegen)
add_executable(run_gpu_codegen ${GPU_CODEGEN_DIR}/run_gpu_codegen.cpp)
target_include_directories(run_gpu_codegen PRIVATE ${GPU_CODEGEN_DIR})
target_compile_features(run_gpu_codegen PRIVATE cxx_std_17)

add_custom_target(run_gpu
  COMMAND $<TARGET_FILE:run_gpu_codegen>
  DEPENDS run_gpu_codegen
  COMMENT "Running GPU code generation pipeline (ParallelDetect→ThreadMap→GPUDialect→SharedMem→NVVM→PTX→Occupancy)")

# ─── P11（14_quantization）：量化与混合精度 ─────────────────────────────────

set(QUANT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/14_quantization)
add_executable(run_quantization ${QUANT_DIR}/run_quantization.cpp)
target_include_directories(run_quantization PRIVATE ${QUANT_DIR})
target_compile_features(run_quantization PRIVATE cxx_std_17)

add_custom_target(run_quant
  COMMAND $<TARGET_FILE:run_quantization>
  DEPENDS run_quantization
  COMMENT "Running Quantization & Mixed-Precision pipeline (Calib→Scale→Fusion→Rewrite→MixedPrec→Summary)")

# ─── P12（15_memory_planning）：内存规划 ───────────────────────────────────

set(MEMPLAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/15_memory_planning)
add_executable(run_memory_planning ${MEMPLAN_DIR}/run_memory_planning.cpp)
target_include_directories(run_memory_planning PRIVATE ${MEMPLAN_DIR})
target_compile_features(run_memory_planning PRIVATE cxx_std_17)

add_custom_target(run_memplan
  COMMAND $<TARGET_FILE:run_memory_planning>
  DEPENDS run_memory_planning
  COMMENT "Running Memory Planning pipeline (Liveness→Interference→OffsetPlan→Reuse→InPlace→Summary)")

# ─── P5（6_stablehlo_passes）：真实 MLIR StableHLO Pass 插件 ──────────────

find_package(MLIR QUIET CONFIG)
if(MLIR_FOUND)
  find_package(LLVM CONFIG REQUIRED)

  # StableHLO include 发现
  set(STABLEHLO_INCLUDE_DIR "" CACHE PATH "Include directory containing stablehlo/")
  if(TARGET MLIRStablehlo)
    get_target_property(_stablehlo_incs MLIRStablehlo INTERFACE_INCLUDE_DIRECTORIES)
    if(_stablehlo_incs)
      list(GET _stablehlo_incs 0 STABLEHLO_INCLUDE_DIR)
    endif()
  endif()
  if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "/usr/local/include/stablehlo/dialect/StablehloOps.h")
    set(STABLEHLO_INCLUDE_DIR "/usr/local/include")
  endif()
  if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "/opt/torch-mlir/externals/stablehlo/stablehlo/dialect/StablehloOps.h")
    set(STABLEHLO_INCLUDE_DIR "/opt/torch-mlir/externals/stablehlo")
  endif()
  if(NOT STABLEHLO_INCLUDE_DIR AND MLIR_DIR)
    get_filename_component(_mlir_prefix "${MLIR_DIR}" PATH)
    get_filename_component(_mlir_prefix "${_mlir_prefix}" PATH)
    get_filename_component(_mlir_prefix "${_mlir_prefix}" PATH)
    if(EXISTS "${_mlir_prefix}/include/stablehlo")
      set(STABLEHLO_INCLUDE_DIR "${_mlir_prefix}/include")
    endif()
    if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "${_mlir_prefix}/tools/mlir/include/stablehlo")
      set(STABLEHLO_INCLUDE_DIR "${_mlir_prefix}/tools/mlir/include")
    endif()
  endif()
  if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "/usr/include/stablehlo/dialect/StablehloOps.h")
    set(STABLEHLO_INCLUDE_DIR "/usr/include")
  endif()
  if(STABLEHLO_INCLUDE_DIR)
    message(STATUS "StableHLO include: ${STABLEHLO_INCLUDE_DIR}")
  else()
    message(WARNING "StableHLO include not found. Set -DSTABLEHLO_INCLUDE_DIR=...")
  endif()

  # StableHLO generated include
  set(STABLEHLO_GENERATED_INCLUDE_DIR "" CACHE PATH "StableHLO build dir for .inc files")
  set(_stablehlo_gen_inc "")
  if(STABLEHLO_GENERATED_INCLUDE_DIR AND EXISTS "${STABLEHLO_GENERATED_INCLUDE_DIR}/stablehlo/dialect/BaseAttrInterfaces.h.inc")
    set(_stablehlo_gen_inc "${STABLEHLO_GENERATED_INCLUDE_DIR}")
  elseif(STABLEHLO_INCLUDE_DIR MATCHES "externals/stablehlo$")
    if(EXISTS "${STABLEHLO_INCLUDE_DIR}/build/stablehlo/dialect/BaseAttrInterfaces.h.inc")
      set(_stablehlo_gen_inc "${STABLEHLO_INCLUDE_DIR}/build")
    elseif(EXISTS "${STABLEHLO_INCLUDE_DIR}/build/include/stablehlo/dialect/BaseAttrInterfaces.h.inc")
      set(_stablehlo_gen_inc "${STABLEHLO_INCLUDE_DIR}/build/include")
    endif()
  elseif(EXISTS "/opt/torch-mlir/externals/stablehlo/build/stablehlo/dialect/BaseAttrInterfaces.h.inc")
    set(_stablehlo_gen_inc "/opt/torch-mlir/externals/stablehlo/build")
  elseif(EXISTS "/opt/torch-mlir/build/stablehlo/stablehlo/dialect/BaseAttrInterfaces.h.inc")
    set(_stablehlo_gen_inc "/opt/torch-mlir/build/stablehlo")
  elseif(EXISTS "/usr/local/include/stablehlo/dialect/BaseAttrInterfaces.h.inc")
    set(_stablehlo_gen_inc "/usr/local/include")
  endif()
  if(_stablehlo_gen_inc)
    message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
  endif()

  # StableHLO 链接库
  set(STABLEHLO_LIB_DIR "" CACHE PATH "Directory containing StableHLO libs")
  set(_stablehlo_libs "")
  if(TARGET MLIRStablehlo)
    set(_stablehlo_libs MLIRStablehlo)
  else()
    if(STABLEHLO_LIB_DIR)
      set(_search_dirs "${STABLEHLO_LIB_DIR}")
    else()
      set(_search_dirs "")
      if(DEFINED ENV{TORCH_MLIR_HOME} AND EXISTS "$ENV{TORCH_MLIR_HOME}/build/lib")
        list(APPEND _search_dirs "$ENV{TORCH_MLIR_HOME}/build/lib")
      endif()
      if(DEFINED ENV{TORCH_MLIR_HOME} AND EXISTS "$ENV{TORCH_MLIR_HOME}/externals/stablehlo/build/lib")
        list(APPEND _search_dirs "$ENV{TORCH_MLIR_HOME}/externals/stablehlo/build/lib")
      endif()
      if(EXISTS "/opt/torch-mlir/build/lib")
        list(APPEND _search_dirs "/opt/torch-mlir/build/lib")
      endif()
      if(EXISTS "/opt/torch-mlir/externals/stablehlo/build/lib")
        list(APPEND _search_dirs "/opt/torch-mlir/externals/stablehlo/build/lib")
      endif()
      if(EXISTS "/usr/local/lib")
        list(APPEND _search_dirs "/usr/local/lib")
      endif()
      if(EXISTS "/usr/lib")
        list(APPEND _search_dirs "/usr/lib")
      endif()
    endif()
    set(_shlo_lib_names StablehloOps StablehloBase StablehloAssemblyFormat StablehloTypeInference StablehloSerialization)
    foreach(_name IN LISTS _shlo_lib_names)
      find_library(_shlo_found_${_name} ${_name} PATHS ${_search_dirs} NO_DEFAULT_PATH)
      if(_shlo_found_${_name})
        list(APPEND _stablehlo_libs "${_shlo_found_${_name}}")
      endif()
    endforeach()
    if(NOT _stablehlo_libs)
      message(FATAL_ERROR "StableHLO: no libs found. Set STABLEHLO_LIB_DIR.")
    endif()
  endif()

  set(PASSES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/6_stablehlo_passes)

  add_definitions(${MLIR_DEFINITIONS})

  add_library(DialectPass SHARED
    ${PASSES_DIR}/ConvBNFusionPattern.cpp
    ${PASSES_DIR}/Passes.cpp
  )
  target_include_directories(DialectPass PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${STABLEHLO_INCLUDE_DIR}
    ${_stablehlo_gen_inc}
    ${PASSES_DIR}
  )
  target_compile_options(DialectPass PRIVATE -fno-rtti)
  target_link_libraries(DialectPass
    -Wl,--whole-archive
    MLIRIR
    MLIRRewrite
    -Wl,--no-whole-archive
    MLIRPass
    ${_stablehlo_libs}
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTransforms
  )

  # mlir-opt 可执行文件
  get_filename_component(_mlir_bin "${MLIR_DIR}/../../bin" ABSOLUTE)
  if(EXISTS "${_mlir_bin}/mlir-opt")
    set(MLIR_OPT_EXE "${_mlir_bin}/mlir-opt")
  else()
    set(MLIR_OPT_EXE "mlir-opt")
  endif()
  message(STATUS "Using mlir-opt: ${MLIR_OPT_EXE}")

  # conv_bn_optimized: Python 导出 → mlir-opt pass
  set(CONV_BN_PYTHON "${Python3_EXECUTABLE}")
  if(NOT CONV_BN_PYTHON)
    set(CONV_BN_PYTHON "python3")
  endif()
  set(CONV_BN_PY "${CMAKE_CURRENT_SOURCE_DIR}/4_torch_to_stablehlo/conv_bn_model.py")
  set(CONV_BN_MODEL_MLIR "${CMAKE_CURRENT_BINARY_DIR}/conv_bn_model.mlir")
  set(CONV_BN_OPTIMIZED_MLIR "${CMAKE_CURRENT_BINARY_DIR}/conv_bn_fusion.mlir")
  configure_file(
    "${PASSES_DIR}/run_mlir_opt.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/run_mlir_opt.sh"
    @ONLY
  )
  add_custom_target(conv_bn_optimized
    COMMAND ${CMAKE_COMMAND} -E echo "Generating conv_bn_model.mlir via conv_bn_model.py ..."
    COMMAND ${_conv_bn_env_cmd} ${CONV_BN_PYTHON} "${CONV_BN_PY}" -o "${CONV_BN_MODEL_MLIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Running mlir-opt --conv-bn-fusion -> conv_bn_fusion.mlir"
    COMMAND sh "${CMAKE_CURRENT_BINARY_DIR}/run_mlir_opt.sh" "${MLIR_OPT_EXE}" "${CONV_BN_MODEL_MLIR}" "${CONV_BN_OPTIMIZED_MLIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS DialectPass
    COMMENT "conv_bn_model.py -> conv_bn_model.mlir -> mlir-opt conv-bn-fusion -> conv_bn_fusion.mlir"
  )
endif()
