#pragma once

#include "static_aligned_allocator.hpp"
#include "types/simd_tags.h"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename T
        , typename std::enable_if_t<std::is_same_v<simd_tag, scalar_tag>>* = nullptr
    >
    T accumulate(const aligned_vector<T>& values)
    {
        T result = 0;
        for (const T r : values)
        {
            result += r;
        }

        return result;
    }
}