# src: ast 与 pass 子目录；ast 提供解释器/LLVM IR 可执行，pass 提供 opt 插件

add_subdirectory(ast)

if(LLVM_CONFIG)
  add_subdirectory(pass)

  # 使用 opt 加载 SimplePass 插件，对 simple_pass.ll 跑 my-peephole 和 my-cfg
  set(PASS_PLUGIN_SO ${CMAKE_BINARY_DIR}/src/pass/SimplePass.so)
  set(PASS_LL_IN ${PROJECT_SOURCE_DIR}/src/pass/simple_pass.ll)
  set(PASS_LL_OUT ${CMAKE_BINARY_DIR}/src/pass/simple_pass_opt.ll)
  add_custom_target(run_simple_pass
    COMMAND ${CMAKE_COMMAND} -E echo "Running opt with SimplePass plugin: my-peephole, my-cfg"
    COMMAND opt -load-pass-plugin=${PASS_PLUGIN_SO} -passes="my-peephole,my-cfg" ${PASS_LL_IN} -S -o ${PASS_LL_OUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Output: ${PASS_LL_OUT}"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS SimplePass
    COMMENT "opt + SimplePass plugin on simple_pass.ll")
endif()
