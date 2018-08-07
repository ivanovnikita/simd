#include "instantiation.h"
#include "../implementation.hpp"
#include "types/avx512f/vector.hpp"

namespace simd::avx512
{
    template <typename T>
    T accumulate(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>& values)
    {
        return simd::detail::accumulate<vector>(values);
    }

    template float accumulate(const std::vector<float, static_aligned_allocator<float, MAX_REQUIRED_ALIGNMENT>>&);
    template double accumulate(const std::vector<double, static_aligned_allocator<double, MAX_REQUIRED_ALIGNMENT>>&);
}