cmake_minimum_required(VERSION 3.10)
project(LangParser)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find ANTLR4 runtime library
find_path(ANTLR4_INCLUDE_DIR
    NAMES antlr4-runtime.h
    PATH_SUFFIXES antlr4-runtime
)

find_library(ANTLR4_LIBRARY
    NAMES antlr4-runtime
)

if(NOT ANTLR4_INCLUDE_DIR)
    message(FATAL_ERROR "ANTLR4 include directory not found. Please install antlr4-cpp-runtime-devel")
endif()

if(NOT ANTLR4_LIBRARY)
    message(FATAL_ERROR "ANTLR4 library not found. Please install antlr4-cpp-runtime-devel")
endif()

message(STATUS "ANTLR4 include directory: ${ANTLR4_INCLUDE_DIR}")
message(STATUS "ANTLR4 library: ${ANTLR4_LIBRARY}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
    ${ANTLR4_INCLUDE_DIR}
)

# Collect all generated source files
file(GLOB GENERATED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/generated/*.cpp
)

# Create a library from the generated parser files
add_library(lang_parser STATIC ${GENERATED_SOURCES})

# Link ANTLR4 runtime library
target_link_libraries(lang_parser ${ANTLR4_LIBRARY})

# Export include directories for parent project
target_include_directories(lang_parser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/generated
    ${ANTLR4_INCLUDE_DIR}
)

# Set compiler flags
target_compile_options(lang_parser PRIVATE ${ANTLR4_CFLAGS_OTHER})
