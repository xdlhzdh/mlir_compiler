# Run MLIR-related targets (conv_bn_model.py + mlir-opt); PASS filters which to
# run. PASS=conv_bn_fusion -> run conv_bn_optimized (generate model.mlir, then
# conv-bn-fusion -> optimized.mlir) PASS=               -> run all MLIR targets
# (currently only conv_bn_optimized) Invoke: cmake -DPASS=conv_bn_fusion
# -DBINARY_DIR=/path/to/build -P run_mlir.cmake From build: make run DOMAIN=mlir
# PASS=conv_bn_fusion

if(NOT BINARY_DIR)
  message(FATAL_ERROR "BINARY_DIR required")
endif()

set(CONV_BN_FUSION_TARGET conv_bn_optimized)
set(GRAPH_TARGET run_graph)
set(LOWERING_TARGET run_lowering)
set(SHLO_OPT_TARGET run_shlo_opt)
set(LINALG_OPT_TARGET run_linalg)
set(BUFFERIZE_TARGET run_buf)
set(SCF_AFFINE_TARGET run_scf)
set(VECTOR_TARGET run_vec)
set(LLVM_LOWER_TARGET run_llvm_lower)
set(GPU_CODEGEN_TARGET run_gpu)
set(QUANTIZATION_TARGET run_quant)
set(MEMPLAN_TARGET run_memplan)
set(ALL_MLIR_TARGETS ${CONV_BN_FUSION_TARGET} ${GRAPH_TARGET} ${LOWERING_TARGET} ${SHLO_OPT_TARGET} ${LINALG_OPT_TARGET} ${BUFFERIZE_TARGET} ${SCF_AFFINE_TARGET} ${VECTOR_TARGET} ${LLVM_LOWER_TARGET} ${GPU_CODEGEN_TARGET} ${QUANTIZATION_TARGET} ${MEMPLAN_TARGET})

function(build_mlir_target name)
  message(STATUS "========== MLIR: ${name} ==========")
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${name}
    WORKING_DIRECTORY ${BINARY_DIR}
    RESULT_VARIABLE rv)
  if(rv)
    message(FATAL_ERROR "MLIR target ${name} failed with ${rv}")
  endif()
endfunction()

function(try_build_mlir_target name out_failed_list)
  message(STATUS "========== MLIR: ${name} ==========")
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${name}
    WORKING_DIRECTORY ${BINARY_DIR}
    RESULT_VARIABLE rv)
  if(rv)
    message(WARNING "MLIR target ${name} failed (exit ${rv}), skipping")
    list(APPEND ${out_failed_list} "${name}")
    set(${out_failed_list} "${${out_failed_list}}" PARENT_SCOPE)
  endif()
endfunction()

if(PASS STREQUAL "conv_bn_fusion")
  build_mlir_target(${CONV_BN_FUSION_TARGET})
elseif(PASS STREQUAL "graph")
  build_mlir_target(${GRAPH_TARGET})
elseif(PASS STREQUAL "lowering")
  build_mlir_target(${LOWERING_TARGET})
elseif(PASS STREQUAL "shlo_opt")
  build_mlir_target(${SHLO_OPT_TARGET})
elseif(PASS STREQUAL "linalg")
  build_mlir_target(${LINALG_OPT_TARGET})
elseif(PASS STREQUAL "bufferize")
  build_mlir_target(${BUFFERIZE_TARGET})
elseif(PASS STREQUAL "scf_affine")
  build_mlir_target(${SCF_AFFINE_TARGET})
elseif(PASS STREQUAL "vector")
  build_mlir_target(${VECTOR_TARGET})
elseif(PASS STREQUAL "llvm_lower")
  build_mlir_target(${LLVM_LOWER_TARGET})
elseif(PASS STREQUAL "gpu_codegen")
  build_mlir_target(${GPU_CODEGEN_TARGET})
elseif(PASS STREQUAL "quantization")
  build_mlir_target(${QUANTIZATION_TARGET})
elseif(PASS STREQUAL "memplan")
  build_mlir_target(${MEMPLAN_TARGET})
elseif(PASS)
  message(WARNING "Unknown PASS=${PASS}; running all MLIR targets")
  set(_failed "")
  foreach(t IN LISTS ALL_MLIR_TARGETS)
    try_build_mlir_target(${t} _failed)
  endforeach()
else()
  set(_failed "")
  foreach(t IN LISTS ALL_MLIR_TARGETS)
    try_build_mlir_target(${t} _failed)
  endforeach()
endif()
if(DEFINED _failed AND _failed)
  string(REPLACE ";" ", " _failed_pretty "${_failed}")
  message(STATUS "Skipped targets (missing optional deps): ${_failed_pretty}")
endif()
