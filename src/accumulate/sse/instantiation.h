#pragma once

#include "static_aligned_allocator.hpp"
#include "types/simd_tags.h"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename T
        , typename std::enable_if_t
        <
            std::is_same_v<simd_tag, sse_tag>
            and std::is_same_v<T, float>
        >* = nullptr
    >
    T accumulate(const aligned_vector<T>& values);
}