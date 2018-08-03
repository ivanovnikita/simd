#include "implementation.h"

#include <immintrin.h>

namespace simd::avx512
{
    float accumulate(const std::vector<float, static_aligned_allocator<float, 64>>& values)
    {
        __m512 a;
        __m512 simd_result = _mm512_setzero_ps();
        for (size_t i = 0; i < values.size(); i += 16)
        {
            a = _mm512_load_ps(&values[i]);
            simd_result = _mm512_add_ps(a, simd_result);
        }

        alignas(64) float result_values[16];
        _mm512_store_ps(result_values, simd_result);

        float result = 0;
        for (const float r : result_values)
        {
            result += r;
        }

        return result;
    }
}