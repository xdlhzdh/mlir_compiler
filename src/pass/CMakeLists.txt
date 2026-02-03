# SimplePass: 构建为 opt 可加载的 .so 插件；pass 实现拆出供 UT 链接
if(NOT LLVM_CONFIG)
  return()
endif()

add_library(SimplePass MODULE
  simple_pass_plugin.cpp
  simple_pass_passes.cpp
)
set_target_properties(SimplePass PROPERTIES
  COMPILE_FLAGS "${LLVM_CXXFLAGS}"
  PREFIX ""
  OUTPUT_NAME "SimplePass"
)
target_include_directories(SimplePass PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SimplePass ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE})

# 静态库供 tests/pass UT 链接（同一 pass 实现）
add_library(SimplePassLib STATIC simple_pass_passes.cpp)
set_target_properties(SimplePassLib PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
target_include_directories(SimplePassLib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SimplePassLib ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE})
