# =============================================================================
# src/mlir/gpu — 与本仓库 MLIR CMake 集成的唯一子目录（见 ../README.md）
#
# - mlir/cpu：无 CMakeLists，仅示例 .mlir / 脚本 / 文档驱动流水线，不 add_subdirectory。
# - 本文件：在 find_package(MLIR) 成功时由 src/CMakeLists.txt 引入；构建 DialectPass
#   插件与 conv_bn_optimized（Python 导出 StableHLO + mlir-opt --conv-bn-fusion）。
# - 「gpu」表示目录用途/后续 GPU 相关实验，不表示本 target 会编译 CUDA kernel。
# =============================================================================

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM CONFIG REQUIRED)

# StableHLO 头文件路径（stablehlo/dialect/StablehloOps.h）：优先从 MLIRStablehlo 目标获取，再试 torch-mlir/llvm 常见路径，否则用 STABLEHLO_INCLUDE_DIR
set(STABLEHLO_INCLUDE_DIR "" CACHE PATH "Include directory containing stablehlo/ (e.g. /opt/torch-mlir/externals/stablehlo or build/tools/mlir/include)")
if(TARGET MLIRStablehlo)
  get_target_property(_stablehlo_incs MLIRStablehlo INTERFACE_INCLUDE_DIRECTORIES)
  if(_stablehlo_incs)
    list(GET _stablehlo_incs 0 STABLEHLO_INCLUDE_DIR)
  endif()
endif()
if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "/usr/local/include/stablehlo/dialect/StablehloOps.h")
  set(STABLEHLO_INCLUDE_DIR "/usr/local/include")
endif()
if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "/opt/torch-mlir/externals/stablehlo/stablehlo/dialect/StablehloOps.h")
  set(STABLEHLO_INCLUDE_DIR "/opt/torch-mlir/externals/stablehlo")
endif()
if(NOT STABLEHLO_INCLUDE_DIR AND MLIR_DIR)
  get_filename_component(_mlir_prefix "${MLIR_DIR}" PATH)
  get_filename_component(_mlir_prefix "${_mlir_prefix}" PATH)
  get_filename_component(_mlir_prefix "${_mlir_prefix}" PATH)
  if(EXISTS "${_mlir_prefix}/include/stablehlo")
    set(STABLEHLO_INCLUDE_DIR "${_mlir_prefix}/include")
  endif()
  if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "${_mlir_prefix}/tools/mlir/include/stablehlo")
    set(STABLEHLO_INCLUDE_DIR "${_mlir_prefix}/tools/mlir/include")
  endif()
endif()
if(NOT STABLEHLO_INCLUDE_DIR AND EXISTS "/usr/include/stablehlo/dialect/StablehloOps.h")
  set(STABLEHLO_INCLUDE_DIR "/usr/include")
endif()
if(STABLEHLO_INCLUDE_DIR)
  message(STATUS "StableHLO include: ${STABLEHLO_INCLUDE_DIR}")
else()
  message(WARNING "StableHLO include not found. If you get stablehlo/dialect/StablehloOps.h not found, run cmake with -DSTABLEHLO_INCLUDE_DIR=/path/to/dir/containing/stablehlo")
endif()

# 使用源码目录时需额外包含“构建生成”的头文件（如 BaseAttrInterfaces.h.inc），否则会报 No such file
# 生成文件通常在 build/stablehlo/dialect/ 或 build/include/stablehlo/dialect/
set(STABLEHLO_GENERATED_INCLUDE_DIR "" CACHE PATH "StableHLO build dir (parent of stablehlo/) for .inc files")
set(_stablehlo_gen_inc "")
if(STABLEHLO_GENERATED_INCLUDE_DIR AND EXISTS "${STABLEHLO_GENERATED_INCLUDE_DIR}/stablehlo/dialect/BaseAttrInterfaces.h.inc")
  set(_stablehlo_gen_inc "${STABLEHLO_GENERATED_INCLUDE_DIR}")
  message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
elseif(STABLEHLO_INCLUDE_DIR MATCHES "externals/stablehlo$")
  if(EXISTS "${STABLEHLO_INCLUDE_DIR}/build/stablehlo/dialect/BaseAttrInterfaces.h.inc")
    set(_stablehlo_gen_inc "${STABLEHLO_INCLUDE_DIR}/build")
    message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
  elseif(EXISTS "${STABLEHLO_INCLUDE_DIR}/build/include/stablehlo/dialect/BaseAttrInterfaces.h.inc")
    set(_stablehlo_gen_inc "${STABLEHLO_INCLUDE_DIR}/build/include")
    message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
  endif()
elseif(EXISTS "/opt/torch-mlir/externals/stablehlo/build/stablehlo/dialect/BaseAttrInterfaces.h.inc")
  set(_stablehlo_gen_inc "/opt/torch-mlir/externals/stablehlo/build")
  message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
elseif(EXISTS "/opt/torch-mlir/build/stablehlo/stablehlo/dialect/BaseAttrInterfaces.h.inc")
  set(_stablehlo_gen_inc "/opt/torch-mlir/build/stablehlo")
  message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
elseif(EXISTS "/usr/local/include/stablehlo/dialect/BaseAttrInterfaces.h.inc")
  set(_stablehlo_gen_inc "/usr/local/include")
  message(STATUS "StableHLO generated include: ${_stablehlo_gen_inc}")
endif()

# 链接 StableHLO：有 target 则用 target；否则用 standalone 安装的库（libStablehloOps.a 等，无 libMLIRStablehlo）
set(STABLEHLO_LIB_DIR "" CACHE PATH "Directory containing StableHLO libs (e.g. /usr/local/lib after 1.2.2 install)")
set(_stablehlo_libs "")
if(TARGET MLIRStablehlo)
  set(_stablehlo_libs MLIRStablehlo)
  message(STATUS "StableHLO link: target MLIRStablehlo")
else()
  if(STABLEHLO_LIB_DIR)
    set(_search_dirs "${STABLEHLO_LIB_DIR}")
  else()
    set(_search_dirs "")
    if(DEFINED ENV{TORCH_MLIR_HOME} AND EXISTS "$ENV{TORCH_MLIR_HOME}/build/lib")
      list(APPEND _search_dirs "$ENV{TORCH_MLIR_HOME}/build/lib")
    endif()
    if(DEFINED ENV{TORCH_MLIR_HOME} AND EXISTS "$ENV{TORCH_MLIR_HOME}/externals/stablehlo/build/lib")
      list(APPEND _search_dirs "$ENV{TORCH_MLIR_HOME}/externals/stablehlo/build/lib")
    endif()
    if(EXISTS "/opt/torch-mlir/build/lib")
      list(APPEND _search_dirs "/opt/torch-mlir/build/lib")
    endif()
    if(EXISTS "/opt/torch-mlir/externals/stablehlo/build/lib")
      list(APPEND _search_dirs "/opt/torch-mlir/externals/stablehlo/build/lib")
    endif()
    if(EXISTS "/usr/local/lib")
      list(APPEND _search_dirs "/usr/local/lib")
    endif()
    if(EXISTS "/usr/lib")
      list(APPEND _search_dirs "/usr/lib")
    endif()
  endif()
  # standalone 安装的库名（1.2.2 ninja install 后）；仅注册 StablehloDialect 时只需 Ops/Base 等
  set(_shlo_lib_names StablehloOps StablehloBase StablehloAssemblyFormat StablehloTypeInference StablehloSerialization)
  foreach(_name IN LISTS _shlo_lib_names)
    find_library(_shlo_found_${_name} ${_name} PATHS ${_search_dirs} NO_DEFAULT_PATH)
    if(_shlo_found_${_name})
      list(APPEND _stablehlo_libs "${_shlo_found_${_name}}")
    endif()
  endforeach()
  if(_stablehlo_libs)
    message(STATUS "StableHLO link: standalone libs (e.g. StablehloOps, StablehloBase, ...)")
  else()
    message(FATAL_ERROR "StableHLO: no MLIRStablehlo target and no standalone libs (StablehloOps, StablehloBase, ...) found. Install StableHLO to /usr/local (docs 1.2.2) or set STABLEHLO_LIB_DIR. See docs 1.2.1.")
  endif()
endif()

add_definitions(${MLIR_DEFINITIONS})
include_directories(
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${STABLEHLO_INCLUDE_DIR}
  ${_stablehlo_gen_inc}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# 输出 libDialectPass.so，供 mlir-opt --load-pass-plugin=./libDialectPass.so 使用
add_library(DialectPass SHARED
  lib/ConvBNFusionPattern.cpp
  lib/Passes.cpp
)

target_include_directories(DialectPass PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${STABLEHLO_INCLUDE_DIR}
  ${_stablehlo_gen_inc}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
)
# 与 MLIR/LLVM 构建一致，避免插件加载时未定义符号 _ZTIN4mlir14RewritePatternE（RTTI typeinfo）
target_compile_options(DialectPass PRIVATE -fno-rtti)

target_link_libraries(DialectPass
  -Wl,--whole-archive
  MLIRIR
  MLIRRewrite
  -Wl,--no-whole-archive
  MLIRPass
  ${_stablehlo_libs}
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
)

# 使用与插件同一 MLIR 构建的 mlir-opt，避免加载 .so 时 ABI/符号不一致
get_filename_component(_mlir_bin "${MLIR_DIR}/../../bin" ABSOLUTE)
if(EXISTS "${_mlir_bin}/mlir-opt")
  set(MLIR_OPT_EXE "${_mlir_bin}/mlir-opt")
  message(STATUS "Using mlir-opt: ${MLIR_OPT_EXE}")
else()
  set(MLIR_OPT_EXE "mlir-opt")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_EXECUTABLE)
  set(CONV_BN_PYTHON "${Python3_EXECUTABLE}")
else()
  set(CONV_BN_PYTHON "python3")
endif()
set(CONV_BN_PY "${CMAKE_CURRENT_SOURCE_DIR}/python/conv_bn_model.py")
set(CONV_BN_MODEL_MLIR "${CMAKE_CURRENT_BINARY_DIR}/conv_bn_model.mlir")
set(CONV_BN_OPTIMIZED_MLIR "${CMAKE_CURRENT_BINARY_DIR}/conv_bn_fusion.mlir")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/run_mlir_opt.sh.in"
  "${CMAKE_CURRENT_BINARY_DIR}/run_mlir_opt.sh"
  @ONLY
)
add_custom_target(conv_bn_optimized
  COMMAND ${CMAKE_COMMAND} -E echo "Generating conv_bn_model.mlir via conv_bn_model.py ..."
  COMMAND ${CONV_BN_PYTHON} "${CONV_BN_PY}" -o "${CONV_BN_MODEL_MLIR}"
  COMMAND ${CMAKE_COMMAND} -E echo "Running mlir-opt --conv-bn-fusion -> conv_bn_fusion.mlir"
  COMMAND sh "${CMAKE_CURRENT_BINARY_DIR}/run_mlir_opt.sh" "${MLIR_OPT_EXE}" "${CONV_BN_MODEL_MLIR}" "${CONV_BN_OPTIMIZED_MLIR}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS DialectPass
  COMMENT "conv_bn_model.py -o conv_bn_model.mlir && mlir-opt conv_bn_model.mlir ... -o conv_bn_fusion.mlir"
)

