# Run ctest; only DOMAIN=ast and DOMAIN=pass supported. DOMAIN=ast  -> ctest -R
# "^V[1-4]_" (ast tests) DOMAIN=pass -> ctest -L pass (pass tests) no DOMAIN ->
# ctest (all) Invoke: cmake -DDOMAIN=ast -DBINARY_DIR=/path/to/build -P
# run_tests.cmake From build: make test [DOMAIN=ast] | make test_ast | make
# test_pass

if(NOT BINARY_DIR)
  set(BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
set(CTEST_WORKING_DIR ${BINARY_DIR})

if(DOMAIN STREQUAL "ast")
  execute_process(
    COMMAND ${CMAKE_CTEST_COMMAND} -R "^V[1-4]_" --output-on-failure
    WORKING_DIRECTORY ${CTEST_WORKING_DIR}
    RESULT_VARIABLE res)
elseif(DOMAIN STREQUAL "pass")
  execute_process(
    COMMAND ${CMAKE_CTEST_COMMAND} -L pass --output-on-failure
    WORKING_DIRECTORY ${CTEST_WORKING_DIR}
    RESULT_VARIABLE res)
elseif(DOMAIN)
  message(FATAL_ERROR "Unsupported DOMAIN=${DOMAIN}; use ast or pass")
else()
  execute_process(
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CTEST_WORKING_DIR}
    RESULT_VARIABLE res)
endif()
if(res)
  message(FATAL_ERROR "ctest failed with ${res}")
endif()
