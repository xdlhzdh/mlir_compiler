# ANTLR4_INCLUDE_DIR / ANTLR4_LIBRARY are set by root CMakeLists.txt
# (this subdirectory is only add_subdirectory'd when both are found)

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
