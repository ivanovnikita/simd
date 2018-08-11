#pragma once

#include <cstdint>

namespace simd
{
    template <int N>
    struct manual_operations_traits
    {
        static constexpr uint8_t adds_per_cycle = N;
    };
}
