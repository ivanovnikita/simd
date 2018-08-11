#pragma once

#include "static_aligned_allocator.hpp"
#include "types/operations_traits.hpp"

namespace simd::detail
{
    template<typename simd_tag, typename T, typename op_traits = operations_traits<simd_tag, T>>
    T accumulate_simd_impl(const aligned_vector<T>& values);
}
