# Test executables (sources in tests/ast/, headers from include/ast via parent include_directories)

find_package(GTest REQUIRED)

add_executable(test_interpreter_v1 ast/test_interpreter_v1.cpp)
target_link_libraries(test_interpreter_v1 GTest::gtest GTest::gtest_main pthread)

add_executable(test_interpreter_v2 ast/test_interpreter_v2.cpp)
target_link_libraries(test_interpreter_v2 GTest::gtest GTest::gtest_main pthread)

add_executable(test_interpreter_v3 ast/test_interpreter_v3.cpp)
target_link_libraries(test_interpreter_v3 GTest::gtest GTest::gtest_main pthread)

add_executable(test_interpreter_v4 ast/test_interpreter_v4.cpp)
target_link_libraries(test_interpreter_v4 GTest::gtest GTest::gtest_main pthread)

# Tests in tests/ast/ get label "ast" (按域跑测：make run_tests DOMAIN=ast)
set(TEST_LABELS "ast")

add_test(NAME V1_Tests COMMAND test_interpreter_v1)
add_test(NAME V2_Tests COMMAND test_interpreter_v2)
add_test(NAME V3_Tests COMMAND test_interpreter_v3)
add_test(NAME V4_Tests COMMAND test_interpreter_v4)
set_tests_properties(V1_Tests V2_Tests V3_Tests V4_Tests PROPERTIES LABELS "${TEST_LABELS}")

include(GoogleTest)
gtest_discover_tests(test_interpreter_v1 PROPERTIES LABELS "${TEST_LABELS}")
gtest_discover_tests(test_interpreter_v2 PROPERTIES LABELS "${TEST_LABELS}")
gtest_discover_tests(test_interpreter_v3 PROPERTIES LABELS "${TEST_LABELS}")
gtest_discover_tests(test_interpreter_v4 PROPERTIES LABELS "${TEST_LABELS}")

# Pass 源码 UT（需 LLVM + SimplePassLib）
if(LLVM_CONFIG)
  add_executable(test_simple_pass pass/test_simple_pass.cpp)
  target_include_directories(test_simple_pass PRIVATE ${PROJECT_SOURCE_DIR}/include)
  set_target_properties(test_simple_pass PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
  target_link_libraries(test_simple_pass GTest::gtest GTest::gtest_main SimplePassLib
    ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE_ASM} ${LLVM_SYSTEM_LIBS} pthread)
  add_test(NAME Pass_SimplePass_Src COMMAND test_simple_pass)
  set_tests_properties(Pass_SimplePass_Src PROPERTIES LABELS "pass")
endif()

# Domain-specific targets (test_ast, test_pass) are defined in root CMakeLists.txt.
# All current tests in tests/ast/ match ^V[1-4]_ and have LABELS "ast".
