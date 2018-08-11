#pragma once

#include "simd_tags.h"

#include <cstdint>

namespace simd
{
    template<typename simd_tag, typename value_type>
    struct operations_traits;

    template <>
    struct operations_traits<sse_tag, float>
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };

    template <>
    struct operations_traits<avx_tag, float>
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };

    template <>
    struct operations_traits<avx_tag, double>
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };

    template <>
    struct operations_traits<avx2_tag, int8_t>
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };

    template <>
    struct operations_traits<avx512f_tag, float>
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };

    template <>
    struct operations_traits<avx512f_tag, double>
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };
}
