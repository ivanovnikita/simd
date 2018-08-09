#pragma once

#include "static_aligned_allocator.hpp"
#include "types/simd_tags.h"

namespace simd
{
    namespace detail
    {
        template <typename T>
        using accumulate_t = T(const aligned_vector<T>&);

        template <typename T>
        extern detail::accumulate_t<T>* const best_available_accumulate;
    }

    template
    <
        typename simd_tag = best_available_tag
        , typename T
        , typename std::enable_if_t<std::is_same_v<simd_tag,  best_available_tag>>* = nullptr
    >
    T accumulate(const aligned_vector<T>& values)
    {
        return detail::best_available_accumulate<T>(values);
    }
}