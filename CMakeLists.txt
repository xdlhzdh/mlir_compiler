cmake_minimum_required(VERSION 3.10)
project(SimpleCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Demo related executable files
add_executable(demo_interpreter_v1 src/demo_interpreter_v1.cpp)
add_executable(demo_interpreter_v2 src/demo_interpreter_v2.cpp)
add_executable(demo_interpreter_v3 src/demo_interpreter_v3.cpp)
add_executable(demo_interpreter_v4 src/demo_interpreter_v4.cpp)

add_subdirectory(antlr)
add_executable(demo_interpreter_antlr src/demo_interpreter_antlr.cpp)
target_link_libraries(demo_interpreter_antlr lang_parser)

add_executable(demo_nfa_dfa src/demo_nfa_dfa.cpp)

# Find LLVM related compilation options and libraries
find_program(LLVM_CONFIG llvm-config)
if(LLVM_CONFIG)
  # Get LLVM compilation and linking flags
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
  message(STATUS "LLVM CXX Flags: ${LLVM_CXXFLAGS}")
  message(STATUS "LLVM LD Flags: ${LLVM_LDFLAGS}")
  message(STATUS "LLVM Libs (core): ${LLVM_LIBS_CORE}")
  message(STATUS "LLVM Libs (jit): ${LLVM_LIBS_JIT}")

  # Create demo_simple_llvm_ir executable
  add_executable(demo_simple_llvm_ir src/demo_simple_llvm_ir.cpp)

  # Set compilation options
  set_target_properties(demo_simple_llvm_ir PROPERTIES COMPILE_FLAGS
    "${LLVM_CXXFLAGS}")

  # Set linking options
  target_link_libraries(demo_simple_llvm_ir ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE})

  # Create demo_simple_llvm_ir_jit executable (JIT version)
  add_executable(demo_simple_llvm_ir_jit src/demo_simple_llvm_ir_jit.cpp)

  # Set compilation options
  set_target_properties(demo_simple_llvm_ir_jit PROPERTIES COMPILE_FLAGS
    "${LLVM_CXXFLAGS}")

  # Set linking options (JIT needs more libraries)
  target_link_libraries(demo_simple_llvm_ir_jit ${LLVM_LDFLAGS}
    ${LLVM_LIBS_JIT})

  # Demo: interpreter_v4 AST -> LLVM IR (needs -fexceptions to override LLVM -fno-exceptions)
  add_executable(demo_interpreter_v4_llvm src/demo_interpreter_v4_llvm.cpp)
  set_target_properties(demo_interpreter_v4_llvm PROPERTIES COMPILE_FLAGS
    "${LLVM_CXXFLAGS} -DUSE_LLVM_CODEGEN -fexceptions")
  target_link_libraries(demo_interpreter_v4_llvm ${LLVM_LDFLAGS}
    ${LLVM_LIBS_CORE})

  # Path to output .ll file: output directory of the project, file name interpreter_v4.ll
  set(OUTPUT_LL_DIR ${PROJECT_SOURCE_DIR}/output)
  set(OUTPUT_LL_FILE ${OUTPUT_LL_DIR}/interpreter_v4.ll)
  set(OUTPUT_LL_SSA_FILE ${OUTPUT_LL_DIR}/interpreter_v4_ssa.ll)
  set(OUTPUT_LL_OPT_FILE ${OUTPUT_LL_DIR}/interpreter_v4_opt.ll)

  # Custom target: generate .ll and execute with lli
  file(MAKE_DIRECTORY ${OUTPUT_LL_DIR})
  set(VERIFY_IR_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/verify_ir.cmake)
  add_custom_target(demo_interpreter_v4_llvm_run
    COMMAND ${CMAKE_COMMAND} -E echo "Generating LLVM IR: ${OUTPUT_LL_FILE}"
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_interpreter_v4_llvm ${OUTPUT_LL_FILE}
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
    DEPENDS demo_interpreter_v4_llvm
    COMMENT "Generate interpreter_v4.ll, verify, generate SSA IR, optimize SSA IR, and execute with lli")
else()
  message(WARNING "llvm-config not found. LLVM demo targets will not be built.")
endif()

# Create custom demo targetUSE_LLVM_CODEGEN
add_custom_target(
  demo
  COMMAND echo "========== Running Demo Interpreter V1 =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_interpreter_v1
  COMMAND echo ""
  COMMAND echo "========== Running Demo Interpreter V2 =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_interpreter_v2
  COMMAND echo ""
  COMMAND echo "========== Running Demo Interpreter V3 =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_interpreter_v3
  COMMAND echo ""
  COMMAND echo "========== Running Demo Interpreter V4 =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_interpreter_v4
  COMMAND echo ""
  COMMAND echo "========== Running Demo ANTLR Interpreter =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_interpreter_antlr
  COMMAND echo ""
  COMMAND echo "========== Running Demo Simple LLVM IR =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_simple_llvm_ir
  COMMAND echo ""
  COMMAND echo "========== Running Demo Simple LLVM JIT =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_simple_llvm_ir_jit
  COMMAND echo ""
  COMMAND echo "========== Running Demo NFA DFA =========="
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/demo_nfa_dfa
  DEPENDS demo_interpreter_v1 demo_interpreter_v2 demo_interpreter_v3 demo_interpreter_v4 demo_interpreter_antlr demo_simple_llvm_ir demo_simple_llvm_ir_jit demo_nfa_dfa
  COMMENT "Running all demos")

# Google Test - using system installed version
find_package(GTest REQUIRED)

enable_testing()

# Test executables
add_executable(test_interpreter_v1 tests/test_interpreter_v1.cpp)
target_link_libraries(test_interpreter_v1 GTest::gtest GTest::gtest_main
  pthread)

add_executable(test_interpreter_v2 tests/test_interpreter_v2.cpp)
target_link_libraries(test_interpreter_v2 GTest::gtest GTest::gtest_main
  pthread)

add_executable(test_interpreter_v3 tests/test_interpreter_v3.cpp)
target_link_libraries(test_interpreter_v3 GTest::gtest GTest::gtest_main
  pthread)

add_executable(test_interpreter_v4 tests/test_interpreter_v4.cpp)
target_link_libraries(test_interpreter_v4 GTest::gtest GTest::gtest_main
  pthread)

# Add tests
add_test(NAME V1_Tests COMMAND test_interpreter_v1)
add_test(NAME V2_Tests COMMAND test_interpreter_v2)
add_test(NAME V3_Tests COMMAND test_interpreter_v3)
add_test(NAME V4_Tests COMMAND test_interpreter_v4)

include(GoogleTest)
gtest_discover_tests(test_interpreter_v1)
gtest_discover_tests(test_interpreter_v2)
gtest_discover_tests(test_interpreter_v3)
gtest_discover_tests(test_interpreter_v4)
