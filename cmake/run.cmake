# Run executables and/or pass targets; only DOMAIN=ast and DOMAIN=pass supported.
#   DOMAIN=ast  -> run only ast (interpreter v1..v4, antlr, nfa_dfa)
#   DOMAIN=pass -> run only pass targets (opt + plugin); PASS=simple_pass to run one plugin
#   DOMAIN=     -> run all (ast + pass when RUN_HAVE_PASS)
# Invoke: cmake -DDOMAIN=pass -DPASS=simple_pass -DBINARY_DIR=/path/to/build -P run.cmake
# From build: make run [DOMAIN=ast] | make run DOMAIN=pass [PASS=simple_pass] | make run_ast | make run_pass

if(NOT BINARY_DIR)
  message(FATAL_ERROR "BINARY_DIR required")
endif()

set(SCRIPT_DIR ${CMAKE_CURRENT_LIST_DIR})

set(AST_RUNS
  run_interpreter_v1 run_interpreter_v2 run_interpreter_v3 run_interpreter_v4
  run_interpreter_antlr run_nfa_dfa)

# Executables from add_subdirectory(src) are in BINARY_DIR/src/
set(EXE_PREFIX "${BINARY_DIR}/src")
function(run_one name)
  set(exe "${EXE_PREFIX}/${name}")
  if(EXISTS "${exe}")
    message(STATUS "========== Running ${name} ==========")
    execute_process(COMMAND "${exe}" WORKING_DIRECTORY ${BINARY_DIR} RESULT_VARIABLE rv)
    if(rv)
      message(FATAL_ERROR "${name} failed with ${rv}")
    endif()
  endif()
endfunction()

# PASS=simple_pass -> run only that plugin; PASS= -> run all plugin pass targets
function(run_pass_domain)
  message(STATUS "========== Run pass (PASS=${PASS}) ==========")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -DPLUGIN=${PASS} -DBINARY_DIR=${BINARY_DIR} -P ${SCRIPT_DIR}/run_pass.cmake
    WORKING_DIRECTORY ${BINARY_DIR}
    RESULT_VARIABLE rv)
  if(rv)
    message(FATAL_ERROR "run_pass.cmake failed with ${rv}")
  endif()
endfunction()

if(DOMAIN STREQUAL "ast")
  foreach(t IN LISTS AST_RUNS)
    run_one(${t})
  endforeach()
elseif(DOMAIN STREQUAL "pass")
  run_pass_domain()
elseif(DOMAIN)
  message(FATAL_ERROR "Unsupported DOMAIN=${DOMAIN}; use ast or pass")
else()
  foreach(t IN LISTS AST_RUNS)
    run_one(${t})
  endforeach()
  if(RUN_HAVE_PASS)
    run_pass_domain()
  endif()
endif()
