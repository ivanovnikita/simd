#pragma once

#include "simd_tags.h"

#include <cstdint>

namespace simd
{
    struct operations_traits
    {
        template <typename simd_tag, typename value_type>
        static const uint8_t adds_per_cycle;
    };
}
