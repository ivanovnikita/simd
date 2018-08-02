#include "operations.h"

#include <xmmintrin.h>

#include <iostream>

namespace simd_sse
{
    float accumulate(const float initial)
    {
        const size_t size = 16384;
        alignas(16) float values[size];

        for (size_t i = 0; i < size; ++i)
        {
            values[i] = i % 2;
        }

        __m128 a;
        __m128 simd_result = _mm_setzero_ps();
        for (size_t i = 0; i < size; i += 4)
        {
            a = _mm_load_ps(values);
            simd_result = _mm_add_ps(a, simd_result);
        }

        alignas(16) float result_values[4];
        _mm_store_ps(result_values, simd_result);

        float result = initial;
        for (const float r : result_values)
        {
            result += r;
        }

        return result;
    }
}