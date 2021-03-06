#pragma once

#include "declaration.hpp"
#include "../simd_implementation_def.hpp"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename value_type
        , typename traits
        , typename std::enable_if_t
        <
            std::is_same_v<simd_tag, avx2_tag>
            and
            (
                std::is_same_v<value_type, int8_t>
                or std::is_same_v<value_type, int16_t>
                or std::is_same_v<value_type, int32_t>
                or std::is_same_v<value_type, int64_t>
            )
        >*
    >
    value_type accumulate(const aligned_vector<value_type>& values)
    {
        return accumulate_simd_impl<simd_tag, value_type, traits>(values);
    }
}