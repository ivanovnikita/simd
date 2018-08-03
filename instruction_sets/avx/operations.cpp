#include "operations.h"

#include <immintrin.h>

namespace simd::avx
{
    float accumulate(const std::vector<float, static_aligned_allocator<float, 64>>& values)
    {
        __m256 a;
        __m256 simd_result = _mm256_setzero_ps();
        for (size_t i = 0; i < values.size(); i += 8)
        {
            a = _mm256_load_ps(&values[i]);
            simd_result = _mm256_add_ps(a, simd_result);
        }

        alignas(32) float result_values[8];
        _mm256_store_ps(result_values, simd_result);

        float result = 0;
        for (const float r : result_values)
        {
            result += r;
        }

        return result;
    }
}