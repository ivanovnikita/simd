#include "operations.h"

#include <immintrin.h>

#include <iostream>

namespace simd_avx
{
    float accumulate(const float initial)
    {
        const size_t size = 16384;
        alignas(32) float values[size];

        for (size_t i = 0; i < size; ++i)
        {
            values[i] = i % 2;
        }

        __m256 a;
        __m256 simd_result = _mm256_setzero_ps();
        for (size_t i = 0; i < size; i += 8)
        {
            a = _mm256_load_ps(values);
            simd_result = _mm256_add_ps(a, simd_result);
        }

        alignas(32) float result_values[8];
        _mm256_store_ps(result_values, simd_result);

        float result = initial;
        for (const float r : result_values)
        {
            result += r;
        }

        return result;
    }
}