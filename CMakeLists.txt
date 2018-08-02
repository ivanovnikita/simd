project(simd)
cmake_minimum_required(VERSION 3.9)

function(sources_in_dir directory result)
    file(GLOB output
        "${directory}/*.cpp"
        "${directory}/*.h"
        "${directory}/*.hpp"
    )
    set(${result} ${output} PARENT_SCOPE)
endfunction()

sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/sse" sse_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/sse2" sse2_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/sse3" sse3_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/ssse3" ssse3_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/sse4.1" sse4.1_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/sse4.2" sse4.2_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/avx" avx_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/avx2" avx2_src)
sources_in_dir("${CMAKE_CURRENT_SOURCE_DIR}/instruction_sets/avx-512" avx-512_src)

add_library(sse OBJECT ${sse_src})
add_library(sse2 OBJECT ${sse2_src})
add_library(sse3 OBJECT ${sse3_src})
add_library(ssse3 OBJECT ${ssse3_src})
add_library(sse4.1 OBJECT ${sse4.1_src})
add_library(sse4.2 OBJECT ${sse4.2_src})
add_library(avx OBJECT ${avx_src})
add_library(avx2 OBJECT ${avx2_src})
add_library(avx-512 OBJECT ${avx-512_src})

set_target_properties(sse PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sse2 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sse3 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ssse3 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sse4.1 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(sse4.2 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(avx PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(avx2 PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(avx-512 PROPERTIES LINKER_LANGUAGE CXX)

target_compile_options(sse PRIVATE -msse)
target_compile_options(sse2 PRIVATE -msse2)
target_compile_options(sse3 PRIVATE -msse3)
target_compile_options(ssse3 PRIVATE -mssse3)
target_compile_options(sse4.1 PRIVATE -msse4.1)
target_compile_options(sse4.2 PRIVATE -msse4.2)
target_compile_options(avx PRIVATE -mavx)
target_compile_options(avx2 PRIVATE -mavx2)
target_compile_options(avx-512 PRIVATE -mavx512f)

sources_in_dir(${CMAKE_CURRENT_SOURCE_DIR} src)
add_executable(${PROJECT_NAME} 
    ${src}
    $<TARGET_OBJECTS:sse>
    $<TARGET_OBJECTS:sse2>
    $<TARGET_OBJECTS:sse3>
    $<TARGET_OBJECTS:ssse3>
    $<TARGET_OBJECTS:sse4.1>
    $<TARGET_OBJECTS:sse4.2>
    $<TARGET_OBJECTS:avx>
    $<TARGET_OBJECTS:avx2>
    $<TARGET_OBJECTS:avx-512>
)
