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
else()
  message(WARNING "llvm-config not found. LLVM demo targets will not be built.")
endif()

# Create custom demo target
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
