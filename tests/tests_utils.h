#pragma once

#include "instrset_detect.hpp"

#include <gtest/gtest.h>

#include <cstdio>

template<typename S, typename V>
struct TypePair
{
    using simd_tag = S;
    using value_type = V;
};

#define SKIP_UNAVAILABLE_TEST(simd_tag)                                         \
    if constexpr (simd::is_simd_tag<simd_tag>())                                \
    {                                                                           \
        if (not simd::has_simd_instrset<simd_tag>())                            \
        {                                                                       \
            const ::testing::TestInfo* const test_info =                        \
                ::testing::UnitTest::GetInstance()->current_test_info();        \
            std::printf                                                         \
            (                                                                   \
                "WARNING: Instr set is not available, skip test %s:%s:%s! \n"   \
                , test_info->test_case_name()                                   \
                , test_info->name()                                             \
                , test_info->type_param() != nullptr                            \
                    ? test_info->type_param()                                   \
                    : ""                                                        \
            );                                                                  \
            SUCCEED();                                                          \
            return;                                                             \
        }                                                                       \
    }