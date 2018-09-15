#pragma once

#include "instrset_detect.hpp"

#include <gtest/gtest.h>

#include <iostream>

template<typename S, typename V>
struct TypePair
{
    using simd_tag = S;
    using value_type = V;
};

#define SKIP_UNAVAILABLE_TEST(simd_tag)                                                     \
    if constexpr (simd::is_simd_tag<simd_tag>())                                            \
    {                                                                                       \
        if (not simd::has_simd_instrset<simd_tag>())                                        \
        {                                                                                   \
            std::cout << "WARNING: Instr set is not available, skip test!" << std::endl;    \
            SUCCEED();                                                                      \
            return;                                                                         \
        }                                                                                   \
    }