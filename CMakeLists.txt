project(simd)
cmake_minimum_required(VERSION 3.10)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    option(USE_LIBC++ "Use libc++?" OFF)
    if (${USE_LIBC++})
        #message()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        set(CONAN_STD_OPTION "CLANG_STD")
    endif()
endif()

include("cmake/cmake-conan/conan.cmake")

conan_cmake_run(
    CONANFILE conanfile.txt
    ${CONAN_STD_OPTION}
    BASIC_SETUP
    CMAKE_TARGETS
    BUILD missing
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
else()
    message(FATAL_ERROR "Unknown value of CMAKE_BUILD_TYPE!")
endif()

set(implementation_types
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
    if(${implementation} STREQUAL "sse")
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

function(add_implementation directory result_src)
    sources_in_dir(${directory} all_sources)
    foreach(implementation_type ${implementation_types})
        sources_in_dir("${directory}/${implementation_type}" src)
        get_implementation_flags(implementation_type flags)
        set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS "${COMMON_CXX_FLAGS} ${flags}")
        list(APPEND all_sources ${src})
    endforeach()
    set(${result_src} ${all_sources} PARENT_SCOPE)
endfunction()

add_implementation(src/accumulate accumulate_src)
add_implementation(src/types types_src)

sources_in_dir(src sources)
add_library(${PROJECT_NAME}
    ${sources}
    ${types_src}
    ${accumulate_src}
)
target_include_directories(${PROJECT_NAME} PUBLIC src)


add_implementation(benchmarks/accumulate accumulate_benchmark_src)
sources_in_dir(benchmarks benchmark_src)
set(benchmarks "benchmark-${PROJECT_NAME}")
add_executable(${benchmarks}
    ${benchmark_src}
    ${accumulate_benchmark_src}
)
target_link_libraries(${benchmarks} ${PROJECT_NAME} CONAN_PKG::Celero)


enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

sources_in_dir(tests tests_src)
add_executable(tests ${tests_src})
target_link_libraries(tests ${PROJECT_NAME} GTest::Main)

gtest_discover_tests(tests)
