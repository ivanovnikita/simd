#pragma once

#include "static_aligned_allocator.hpp"

namespace simd
{
    namespace detail
    {
        template <typename T>
        using accumulate_t = T(const aligned_vector<T>&);

        template <typename T>
        extern detail::accumulate_t<T>* const best_available_accumulate;
    }

    template <typename T>
    T accumulate(const aligned_vector<T>& values)
    {
        return detail::best_available_accumulate<T>(values);
    }
}