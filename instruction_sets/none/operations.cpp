#include "operations.h"

#include <iostream>

namespace simd_none
{
    float accumulate(const float initial)
    {
        const size_t size = 16384;
        float values[size];

        for (size_t i = 0; i < size; ++i)
        {
            values[i] = i % 2;
        }

        float result = initial;
        for (const float r : values)
        {
            result += r;
        }

        return result;
    }
}