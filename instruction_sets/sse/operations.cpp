#include "operations.h"

#include <xmmintrin.h>

namespace simd::sse
{
    float accumulate(const std::vector<float, static_aligned_allocator<float, 64>>& values)
    {
        __m128 a;
        __m128 simd_result = _mm_setzero_ps();
        for (size_t i = 0; i < values.size(); i += 4)
        {
            a = _mm_load_ps(&values[i]);
            simd_result = _mm_add_ps(a, simd_result);
        }

        alignas(16) float result_values[4];
        _mm_store_ps(result_values, simd_result);

        float result = 0;
        for (const float r : result_values)
        {
            result += r;
        }

        return result;
    }
}