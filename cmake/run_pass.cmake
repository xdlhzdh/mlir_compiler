# Run pass-related targets (opt + plugin); PLUGIN filters which to run.
# PLUGIN=simple_pass  -> run only run_simple_pass (SimplePass plugin) PLUGIN= ->
# run all plugin pass targets Invoke: cmake -DPLUGIN=simple_pass
# -DBINARY_DIR=/path/to/build -P run_pass.cmake From build: make pass
# [PLUGIN=simple_pass]

if(NOT BINARY_DIR)
  message(FATAL_ERROR "BINARY_DIR required")
endif()

set(SIMPLE_PASS_TARGET run_simple_pass)
set(RTB_PASS_TARGET run_remove_trivial_block)
set(RTL_PASS_TARGET run_remove_trivial_loop)
set(RZT_PASS_TARGET run_remove_zero_trip_loop)
set(ALL_PASS_TARGETS ${SIMPLE_PASS_TARGET} ${RTB_PASS_TARGET}
                     ${RTL_PASS_TARGET} ${RZT_PASS_TARGET})

function(build_pass_target name)
  message(STATUS "========== Pass: ${name} ==========")
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${name}
    WORKING_DIRECTORY ${BINARY_DIR}
    RESULT_VARIABLE rv)
  if(rv)
    message(FATAL_ERROR "Pass ${name} failed with ${rv}")
  endif()
endfunction()

if(PLUGIN STREQUAL "simple_pass")
  build_pass_target(${SIMPLE_PASS_TARGET})
elseif(PLUGIN STREQUAL "remove_trivial_block")
  build_pass_target(${RTB_PASS_TARGET})
elseif(PLUGIN STREQUAL "remove_trivial_loop")
  build_pass_target(${RTL_PASS_TARGET})
elseif(PLUGIN STREQUAL "remove_zero_trip_loop")
  build_pass_target(${RZT_PASS_TARGET})
elseif(PLUGIN)
  message(WARNING "Unknown PLUGIN=${PLUGIN}; running all plugin pass targets")
  foreach(t IN LISTS ALL_PASS_TARGETS)
    build_pass_target(${t})
  endforeach()
else()
  foreach(t IN LISTS ALL_PASS_TARGETS)
    build_pass_target(${t})
  endforeach()
endif()
