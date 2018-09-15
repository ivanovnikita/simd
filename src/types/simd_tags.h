#pragma once

#include <type_traits>

namespace simd
{
    struct avx512f_tag {};
    struct avx2_tag {};
    struct avx_tag {};
    struct sse_tag {};
    struct sse2_tag {};
    struct scalar_tag {};
    struct best_available_tag {};

    template <typename simd_tag>
    constexpr bool is_simd_tag()
    {
        if constexpr
        (
            std::is_same_v<simd_tag, sse_tag>
            or std::is_same_v<simd_tag, sse2_tag>
            or std::is_same_v<simd_tag, avx_tag>
            or std::is_same_v<simd_tag, avx2_tag>
            or std::is_same_v<simd_tag, avx512f_tag>
        )
        {
            return true;
        }

        return false;
    }
}
