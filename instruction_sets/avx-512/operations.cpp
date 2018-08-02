#include "operations.h"

#include <immintrin.h>

#include <iostream>

namespace simd_avx512
{
    void do_stuff()
    {
        alignas(64) const float values[16] =
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f
        };

        std::cout << alignof(values) << std::endl;

        __m512 a = _mm512_load_ps(values);
        __m512 b = _mm512_load_ps(values);
        __m512 result = _mm512_add_ps(a, b);

        alignas(64) float result_values[16] = {};
        _mm512_store_ps(result_values, result);

        for (const float value : result_values)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}