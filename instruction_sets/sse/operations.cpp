#include "operations.h"

#include <xmmintrin.h>

#include <iostream>

namespace simd_sse
{
    void do_stuff()
    {
        alignas(16) const float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        __m128 a = _mm_load_ps(values);
        __m128 b = _mm_load_ps(values);
        __m128 result = _mm_add_ps(a, b);

        alignas(16) float result_values[4] = {};
        _mm_store_ps(result_values, result);

        for (const float value : result_values)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}