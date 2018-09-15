#pragma once

#include <type_traits>
#include "instrset_detect.h"
#include "types/simd_tags.h"

namespace simd
{
    template <typename simd_tag>
    bool has_simd_instrset() noexcept
    {
        if constexpr (std::is_same_v<simd_tag, sse_tag>)
        {
            return has_sse();
        }
        else if constexpr (std::is_same_v<simd_tag, avx_tag>)
        {
            return has_avx();
        }
        else if constexpr (std::is_same_v<simd_tag, avx2_tag>)
        {
            return has_avx2();
        }
        else if constexpr (std::is_same_v<simd_tag, avx512f_tag>)
        {
            return has_avx512f();
        }
    }
}