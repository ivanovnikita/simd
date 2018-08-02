#include "operations.h"

#include <immintrin.h>

#include <iostream>

namespace simd_avx
{
    void do_stuff()
    {
        alignas(32) const float values[8] =
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f
        };

        __m256 a = _mm256_load_ps(values);
        __m256 b = _mm256_load_ps(values);
        __m256 result = _mm256_add_ps(a, b);

        alignas(32) float result_values[8];
        _mm256_store_ps(result_values, result);

        for (const float value : result_values)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}