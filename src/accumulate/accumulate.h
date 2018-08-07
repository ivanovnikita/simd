#pragma once

#include "static_aligned_allocator.hpp"

#include <vector>

namespace simd
{
    namespace detail
    {
        template<typename T>
        using accumulate_t = T(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>&);

        template <typename T>
        extern detail::accumulate_t<T>* const accumulate;
    }

    template <typename T>
    T accumulate(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>& values)
    {
        return detail::accumulate<T>(values);
    }
}