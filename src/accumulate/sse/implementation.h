#pragma once

#include "static_aligned_allocator.hpp"

#include <vector>

namespace simd::sse
{
    float accumulate(const std::vector<float, static_aligned_allocator<float, 64>>& values);
}