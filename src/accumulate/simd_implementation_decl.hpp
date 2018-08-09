#pragma once

#include "static_aligned_allocator.hpp"

namespace simd::detail
{
    template<typename simd_tag, typename T>
    T accumulate_simd_impl(const aligned_vector<T>& values);
}
