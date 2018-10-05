#pragma once

#include "simd_tags.h"

#include <cstdint>

namespace simd
{
    template<typename simd_tag, typename value_type>
    struct operations_traits
    {
        static constexpr uint8_t adds_per_cycle = 4;
    };
}
