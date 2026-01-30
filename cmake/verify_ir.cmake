if(NOT DEFINED INPUT_FILE)
  message(FATAL_ERROR "INPUT_FILE is not set for verify_ir.cmake")
endif()

execute_process(
  COMMAND opt -passes=verify -disable-output ${INPUT_FILE}
  RESULT_VARIABLE VERIFY_RESULT
  OUTPUT_VARIABLE VERIFY_STDOUT
  ERROR_VARIABLE VERIFY_STDERR
)

if(VERIFY_RESULT EQUAL 0)
  message(STATUS "Verification result: OK")
else()
  message(STATUS "Verification result: FAILED")
  if(NOT VERIFY_STDERR STREQUAL "")
    message(STATUS "${VERIFY_STDERR}")
  endif()
  message(FATAL_ERROR "opt verify failed for: ${INPUT_FILE}")
endif()
