#include "operations.h"

namespace simd::none
{
    float accumulate(const std::vector<float, static_aligned_allocator<float, 64>>& values)
    {
        float result = 0;
        for (const float r : values)
        {
            result += r;
        }

        return result;
    }
}