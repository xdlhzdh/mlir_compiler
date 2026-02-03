# AST 相关可执行与 run 目标（解释器 v1–v4、antlr、nfa_dfa、LLVM IR 等）

add_executable(run_interpreter_v1 run_interpreter_v1.cpp)
add_executable(run_interpreter_v2 run_interpreter_v2.cpp)
add_executable(run_interpreter_v3 run_interpreter_v3.cpp)
add_executable(run_interpreter_v4 run_interpreter_v4.cpp)

add_executable(run_interpreter_antlr run_interpreter_antlr.cpp)
target_link_libraries(run_interpreter_antlr lang_parser)

add_executable(run_nfa_dfa run_nfa_dfa.cpp)

# LLVM 相关（使用根 CMakeLists 提供的 LLVM_*）
if(LLVM_CONFIG)
  add_executable(run_simple_llvm_ir run_simple_llvm_ir.cpp)
  set_target_properties(run_simple_llvm_ir PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
  target_link_libraries(run_simple_llvm_ir ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE})

  add_executable(run_simple_llvm_ir_jit run_simple_llvm_ir_jit.cpp)
  set_target_properties(run_simple_llvm_ir_jit PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
  target_link_libraries(run_simple_llvm_ir_jit ${LLVM_LDFLAGS} ${LLVM_LIBS_JIT})

  add_executable(run_interpreter_v4_llvm run_interpreter_v4_llvm.cpp)
  set_target_properties(run_interpreter_v4_llvm PROPERTIES COMPILE_FLAGS
    "${LLVM_CXXFLAGS} -DUSE_LLVM_CODEGEN -fexceptions")
  target_link_libraries(run_interpreter_v4_llvm ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE})

  set(OUTPUT_LL_DIR ${PROJECT_SOURCE_DIR}/output)
  set(OUTPUT_LL_FILE ${OUTPUT_LL_DIR}/interpreter_v4.ll)
  set(OUTPUT_LL_SSA_FILE ${OUTPUT_LL_DIR}/interpreter_v4_ssa.ll)
  set(OUTPUT_LL_OPT_FILE ${OUTPUT_LL_DIR}/interpreter_v4_opt.ll)
  file(MAKE_DIRECTORY ${OUTPUT_LL_DIR})
  set(VERIFY_IR_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/verify_ir.cmake)
  add_custom_target(run_interpreter_v4_llvm_run
    COMMAND ${CMAKE_COMMAND} -E echo "Generating LLVM IR: ${OUTPUT_LL_FILE}"
    COMMAND $<TARGET_FILE:run_interpreter_v4_llvm> ${OUTPUT_LL_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying LLVM IR"
    COMMAND ${CMAKE_COMMAND} -DINPUT_FILE=${OUTPUT_LL_FILE} -P ${VERIFY_IR_SCRIPT}
    COMMAND ${CMAKE_COMMAND} -E echo "Generating SSA IR: ${OUTPUT_LL_SSA_FILE}"
    COMMAND opt -passes=mem2reg ${OUTPUT_LL_FILE} -S -o ${OUTPUT_LL_SSA_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying SSA IR"
    COMMAND ${CMAKE_COMMAND} -DINPUT_FILE=${OUTPUT_LL_SSA_FILE} -P ${VERIFY_IR_SCRIPT}
    COMMAND ${CMAKE_COMMAND} -E echo "Optimizing SSA IR: ${OUTPUT_LL_OPT_FILE}"
    COMMAND opt -passes=mem2reg,instcombine,simplifycfg ${OUTPUT_LL_SSA_FILE} -S -o ${OUTPUT_LL_OPT_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying Optimized SSA IR: ${OUTPUT_LL_OPT_FILE}"
    COMMAND ${CMAKE_COMMAND} -DINPUT_FILE=${OUTPUT_LL_OPT_FILE} -P ${VERIFY_IR_SCRIPT}
    COMMAND ${CMAKE_COMMAND} -E echo "Running lli on Optimized SSA IR: ${OUTPUT_LL_OPT_FILE}"
    COMMAND lli ${OUTPUT_LL_OPT_FILE}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS run_interpreter_v4_llvm
    COMMENT "Generate interpreter_v4.ll, verify, SSA/opt, and run lli")
endif()
