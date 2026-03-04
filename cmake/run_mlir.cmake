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
set(ALL_MLIR_TARGETS ${CONV_BN_FUSION_TARGET})

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

if(PASS STREQUAL "conv_bn_fusion")
  build_mlir_target(${CONV_BN_FUSION_TARGET})
elseif(PASS)
  message(WARNING "Unknown PASS=${PASS}; running all MLIR targets")
  foreach(t IN LISTS ALL_MLIR_TARGETS)
    build_mlir_target(${t})
  endforeach()
else()
  foreach(t IN LISTS ALL_MLIR_TARGETS)
    build_mlir_target(${t})
  endforeach()
endif()
