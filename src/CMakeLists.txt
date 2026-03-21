# src: ast、pass 与可选的 mlir/gpu（Conv+BN fusion 等）子目录
#
# MLIR 说明（详见 src/mlir/README.md）：
#   - mlir/gpu — 仅此目录参与 CMake：MLIR 找到则 add_subdirectory(mlir/gpu)。
#   - mlir/cpu — 无 CMakeLists，仅为 Linalg/LLVM 等手工流水线与示例 IR，不构建。

add_subdirectory(ast)

find_package(MLIR QUIET CONFIG)
if(MLIR_FOUND)
  add_subdirectory(mlir/gpu)
endif()

if(LLVM_CONFIG)
  add_subdirectory(pass)

  # 使用 opt 加载 SimplePass 插件，对 simple_pass.ll 跑 my-peephole 和 my-cfg
  set(PASS_PLUGIN_SO ${CMAKE_BINARY_DIR}/src/pass/SimplePass.so)
  set(PASS_LL_IN ${PROJECT_SOURCE_DIR}/src/pass/simple_pass.ll)
  set(PASS_LL_OUT ${CMAKE_BINARY_DIR}/src/pass/simple_pass_opt.ll)
  add_custom_target(run_simple_pass
    COMMAND ${CMAKE_COMMAND} -E echo "Running opt with SimplePass plugin: my-peephole, my-cfg"
    COMMAND opt -load-pass-plugin=${PASS_PLUGIN_SO} -passes="my-peephole,my-cfg" ${PASS_LL_IN} -S -o ${PASS_LL_OUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Output: ${PASS_LL_OUT}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS SimplePass
    COMMENT "opt + SimplePass plugin on simple_pass.ll")

  # RemoveTrivialBlockPass: 对 remove_trivial_block.ll 跑 my-remove-trivial-block
  set(RTB_LL_IN ${PROJECT_SOURCE_DIR}/src/pass/remove_trivial_block.ll)
  set(RTB_LL_OUT ${CMAKE_BINARY_DIR}/src/pass/remove_trivial_block_opt.ll)
  add_custom_target(run_remove_trivial_block
    COMMAND ${CMAKE_COMMAND} -E echo "Running opt with RemoveTrivialBlockPass"
    COMMAND opt -load-pass-plugin=${PASS_PLUGIN_SO} -passes="my-remove-trivial-block" ${RTB_LL_IN} -S -o ${RTB_LL_OUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Output: ${RTB_LL_OUT}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS SimplePass
    COMMENT "opt + RemoveTrivialBlockPass on remove_trivial_block.ll")

  # RemoveTrivialLoopPass: 对 remove_trivial_loop.ll 跑 my-remove-trivial-loop
  set(RTL_LL_IN ${PROJECT_SOURCE_DIR}/src/pass/remove_trivial_loop.ll)
  set(RTL_LL_OUT ${CMAKE_BINARY_DIR}/src/pass/remove_trivial_loop_opt.ll)
  add_custom_target(run_remove_trivial_loop
    COMMAND ${CMAKE_COMMAND} -E echo "Running opt with RemoveTrivialLoopPass"
    COMMAND opt -load-pass-plugin=${PASS_PLUGIN_SO} -passes="my-remove-trivial-loop" ${RTL_LL_IN} -S -o ${RTL_LL_OUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Output: ${RTL_LL_OUT}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS SimplePass
    COMMENT "opt + RemoveTrivialLoopPass on remove_trivial_loop.ll")

  # RemoveZeroTripLoopPass: 对 remove_zero_trip_loop.ll 跑 my-remove-zero-trip-loop
  set(RZT_LL_IN ${PROJECT_SOURCE_DIR}/src/pass/remove_zero_trip_loop.ll)
  set(RZT_LL_OUT ${CMAKE_BINARY_DIR}/src/pass/remove_zero_trip_loop_opt.ll)
  add_custom_target(run_remove_zero_trip_loop
    COMMAND ${CMAKE_COMMAND} -E echo "Running opt with RemoveZeroTripLoopPass"
    COMMAND opt -load-pass-plugin=${PASS_PLUGIN_SO} -passes="my-remove-zero-trip-loop" ${RZT_LL_IN} -S -o ${RZT_LL_OUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Output: ${RZT_LL_OUT}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS SimplePass
    COMMENT "opt + RemoveZeroTripLoopPass on remove_zero_trip_loop.ll")
endif()
