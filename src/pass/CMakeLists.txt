# SimplePass: 构建为 opt 可加载的 .so 插件；pass 实现拆出供 UT 链接
if(NOT LLVM_CONFIG)
  return()
endif()

# Plugin (MODULE) 必须 **不链接** LLVM 静态库，否则 opt 加载时会触发
# CommandLine option 双重注册（"Option XXX registered more than once"）。
# 头文件通过 LLVM_CXXFLAGS 提供；符号在加载时由 host opt 解析（-undefined dynamic_lookup
# 在 macOS 上隐式生效；Linux 上 .so 本就允许未解析符号）。
add_library(SimplePass MODULE
  remove_trivial_block_passes.cpp
  remove_trivial_loop_passes.cpp
  remove_zero_trip_loop_passes.cpp
  simple_pass_plugin.cpp
  simple_pass_passes.cpp
)
set_target_properties(SimplePass PROPERTIES
  COMPILE_FLAGS "${LLVM_CXXFLAGS}"
  PREFIX ""
  OUTPUT_NAME "SimplePass"
)
target_include_directories(SimplePass PRIVATE ${PROJECT_SOURCE_DIR}/include)
# 不链接 LLVM 库；仅传递 -fno-rtti 等 ldflags（如果有的话）。
# 关键：不要把 ${LLVM_LIBS_CORE} 加进来。

# 静态库供 tests/pass UT 链接（独立可执行文件，需要完整 LLVM 库）
add_library(SimplePassLib STATIC simple_pass_passes.cpp)
set_target_properties(SimplePassLib PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")
target_include_directories(SimplePassLib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SimplePassLib ${LLVM_LDFLAGS} ${LLVM_LIBS_CORE})
