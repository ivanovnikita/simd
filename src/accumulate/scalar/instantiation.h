#pragma once

#include "static_aligned_allocator.hpp"

#include <vector>

namespace simd::scalar
{
    template <typename T>
    T accumulate(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>& values);
}