project(simd)
cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
else()
    message(FATAL_ERROR "Unknown value of CMAKE_BUILD_TYPE!")
endif()

set(implementation_types
    default
    sse
    sse2
    sse3
    ssse3
    sse4.1
    sse4.2
    avx
    avx2
    avx512f
)

function(get_implementation_flags implementation result_flags)
    if(${implementation} STREQUAL "default")
        set(flags "-march=native")
    elseif(${implementation} STREQUAL "sse")
        set(flags "-msse")
    elseif(${implementation} STREQUAL "sse2")
        set(flags "-msse2")
    elseif(${implementation} STREQUAL "sse3")
        set(flags "-msse3")
    elseif(${implementation} STREQUAL "ssse3")
        set(flags "-mssse3")
    elseif(${implementation} STREQUAL "sse4.1")
        set(flags "-msse4.1")
    elseif(implementation STREQUAL "sse4.2")
        set(flags "-msse4.2")
    elseif(${implementation} STREQUAL "avx")
        set(flags "-mavx")
    elseif(${implementation} STREQUAL "avx2")
        set(flags "-mavx2")
    elseif(${implementation} STREQUAL "avx512f")
        set(flags "-mavx512f")
    endif()
    set(${result_flags} ${flags} PARENT_SCOPE)
endfunction()

function(sources_in_dir directory result)
    file(GLOB output
        "${directory}/*.cpp"
        "${directory}/*.h"
        "${directory}/*.hpp"
    )
    set(${result} ${output} PARENT_SCOPE)
endfunction()

function(add_algorithm directory result_src)
    set(all_sources "")
    foreach(implementation_type ${implementation_types})
        sources_in_dir("${directory}/${implementation_type}" src)
        get_implementation_flags(implementation_type flags)
        set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${flags}")
        list(APPEND all_sources ${src})
    endforeach()
    set(${result_src} ${all_sources} PARENT_SCOPE)
endfunction()

add_algorithm("${CMAKE_CURRENT_SOURCE_DIR}/accumulate" accumulate_src)

sources_in_dir(${CMAKE_CURRENT_SOURCE_DIR} src)
add_executable(${PROJECT_NAME}
    ${src}
    ${accumulate_src}
)

target_link_libraries(${PROJECT_NAME} -lbenchmark)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})