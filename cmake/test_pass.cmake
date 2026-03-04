# 运行 SimplePass 并校验输出 IR（用于 add_test Pass_SimplePass） 用法: cmake
# -DBINARY_DIR=/path/to/build -DSOURCE_DIR=/path/to/source -P test_pass.cmake

if(NOT BINARY_DIR OR NOT SOURCE_DIR)
  message(FATAL_ERROR "BINARY_DIR and SOURCE_DIR required")
endif()

set(OPT_OUT "${BINARY_DIR}/src/pass/simple_pass_opt.ll")
set(PLUGIN_SO "${BINARY_DIR}/src/pass/SimplePass.so")
set(LL_IN "${SOURCE_DIR}/src/pass/simple_pass.ll")

# 先构建 run_simple_pass 生成 simple_pass_opt.ll
execute_process(
  COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target run_simple_pass
  WORKING_DIRECTORY ${BINARY_DIR}
  RESULT_VARIABLE rv)
if(rv)
  message(FATAL_ERROR "run_simple_pass failed: ${rv}")
endif()

if(NOT EXISTS "${OPT_OUT}")
  message(FATAL_ERROR "Expected output missing: ${OPT_OUT}")
endif()

file(READ "${OPT_OUT}" OPT_CONTENT)

# 优化后应无 "add i32 %x, 0"（已被 peephole 消掉）
string(FIND "${OPT_CONTENT}" "add i32 %x, 0" BAD)
if(NOT BAD EQUAL -1)
  message(FATAL_ERROR "Peephole failed: output still contains 'add i32 %x, 0'")
endif()

# 优化后应有 "ret i32 %x"
string(FIND "${OPT_CONTENT}" "ret i32 %x" GOOD)
if(GOOD EQUAL -1)
  message(FATAL_ERROR "Expected 'ret i32 %x' not found in ${OPT_OUT}")
endif()

message(STATUS "Pass_SimplePass: output IR verified")
