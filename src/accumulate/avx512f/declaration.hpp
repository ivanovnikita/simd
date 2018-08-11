#pragma once

#include "static_aligned_allocator.hpp"
#include "types/operations_traits.hpp"
#include "types/simd_tags.h"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename value_type
        , typename traits = operations_traits<simd_tag, value_type>
        , typename std::enable_if_t
        <
            std::is_same_v<simd_tag, avx512f_tag>
            and
            (
                std::is_same_v<value_type, float>
                or std::is_same_v<value_type, double>
            )
        >* = nullptr
    >
    value_type accumulate(const aligned_vector<value_type>& values);
}