cmake_minimum_required(VERSION 3.10)
project(SimpleCompiler)
# Usage (from build dir):
#   make run | run_ast | run_pass | run DOMAIN=ast | run DOMAIN=pass
#   make test (全量) | make run_tests DOMAIN=ast|pass (按域) | test_ast | test_pass

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Headers in include/ast (used by both src and tests)
include_directories(${PROJECT_SOURCE_DIR}/include/ast)

# LLVM (optional; src/CMakeLists.txt uses these variables)
find_program(LLVM_CONFIG llvm-config)
if(LLVM_CONFIG)
  execute_process(
    COMMAND ${LLVM_CONFIG} --cxxflags
    OUTPUT_VARIABLE LLVM_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND ${LLVM_CONFIG} --ldflags
    OUTPUT_VARIABLE LLVM_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND ${LLVM_CONFIG} --libs core
    OUTPUT_VARIABLE LLVM_LIBS_CORE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND ${LLVM_CONFIG} --libs core orcjit native
    OUTPUT_VARIABLE LLVM_LIBS_JIT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "LLVM found: ${LLVM_CONFIG}")
else()
  set(LLVM_CONFIG "")
  message(WARNING "llvm-config not found. LLVM run targets will not be built.")
endif()

enable_testing()
add_subdirectory(antlr)
add_subdirectory(src)
add_subdirectory(tests)

# Pass UT: script test (opt + plugin)；C++ UT (pass 源码) 在 tests/CMakeLists.txt
if(LLVM_CONFIG)
  add_test(NAME Pass_SimplePass
    COMMAND ${CMAKE_COMMAND} -DBINARY_DIR=${CMAKE_BINARY_DIR} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
      -P ${PROJECT_SOURCE_DIR}/cmake/test_pass.cmake)
  set_tests_properties(Pass_SimplePass PROPERTIES LABELS "pass")
endif()

# RUN_HAVE_PASS: set when LLVM so run.cmake runs pass on "make run"
set(RUN_HAVE_PASS 0)
if(LLVM_CONFIG)
  set(RUN_HAVE_PASS 1)
endif()

# run: DOMAIN=ast|pass|(empty=all), PASS=simple_pass when DOMAIN=pass
add_custom_target(run
  COMMAND ${PROJECT_SOURCE_DIR}/cmake/run_run.sh
    ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/run.cmake
    "\$(DOMAIN)" "\$(PASS)" ${RUN_HAVE_PASS}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS run_interpreter_v1 run_interpreter_v2 run_interpreter_v3 run_interpreter_v4
          run_interpreter_antlr run_nfa_dfa
  COMMENT "Run (make run [DOMAIN=ast|pass] [PASS=simple_pass])")
add_custom_target(run_ast
  COMMAND ${CMAKE_COMMAND} -DDOMAIN=ast -DBINARY_DIR=${CMAKE_BINARY_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/run.cmake
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS run_interpreter_v1 run_interpreter_v2 run_interpreter_v3 run_interpreter_v4
          run_interpreter_antlr run_nfa_dfa
  COMMENT "Run ast only")
if(LLVM_CONFIG)
  add_custom_target(run_pass
    COMMAND ${CMAKE_COMMAND} -DDOMAIN=pass -DBINARY_DIR=${CMAKE_BINARY_DIR}
      -P ${PROJECT_SOURCE_DIR}/cmake/run.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS run_simple_pass
    COMMENT "Run pass only")
endif()

# 修补 Makefile 的 test 目标使 make test DOMAIN=pass|ast 生效；run 依赖之以便首次 make run 后即生效
set(CTEST_WRAPPER "${PROJECT_SOURCE_DIR}/cmake/ctest_wrapper.sh")
add_custom_target(patch_test_domain
  COMMAND sh ${CTEST_WRAPPER} --patch ${CMAKE_BINARY_DIR}/Makefile ${CTEST_WRAPPER}
  COMMENT "Patch test target for make test DOMAIN=ast|pass (idempotent)")
add_dependencies(run patch_test_domain)

# test 按域：make run_tests DOMAIN=ast|pass 或 make test_ast / test_pass；打补丁后 make test DOMAIN=pass 也生效
add_custom_target(run_tests
  COMMAND ${PROJECT_SOURCE_DIR}/cmake/run_tests.sh
    ${CMAKE_COMMAND} ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/run_tests.cmake
    "\$(DOMAIN)"
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Run tests by domain (make run_tests [DOMAIN=ast|pass])")
add_custom_target(test_ast
  COMMAND ${CMAKE_COMMAND} -DDOMAIN=ast -DBINARY_DIR=${CMAKE_BINARY_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/run_tests.cmake
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Test ast only")
add_custom_target(test_pass
  COMMAND ${CMAKE_COMMAND} -DDOMAIN=pass -DBINARY_DIR=${CMAKE_BINARY_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/run_tests.cmake
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Test pass only")
