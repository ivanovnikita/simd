#pragma once

#include "static_aligned_allocator.hpp"

namespace simd::scalar
{
    template <typename T>
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