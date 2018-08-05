#include "implementation.h"
#include "types/avx/vector.hpp"

namespace simd::avx
{
    template <typename T>
    T accumulate(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>& values)
    {
        constexpr uint8_t step = vector<T>::capacity;

        vector<T> simd_result;
        simd_result.setzero_p();

        size_t i = step;
        for (; i < values.size(); i += step)
        {
            simd_result += vector<T>(&values[i - step]);
        }

        alignas(vector<T>::required_alignment) T result_values[vector<T>::capacity];
        simd_result.store_p(result_values);

        i -= step;
        T result = 0;
        for (; i < values.size(); ++i)
        {
            result += values[i];
        }

        for (const T r : result_values)
        {
            result += r;
        }

        return result;
    }

    template float accumulate(const std::vector<float, static_aligned_allocator<float, MAX_REQUIRED_ALIGNMENT>>&);
    template double accumulate(const std::vector<double, static_aligned_allocator<double, MAX_REQUIRED_ALIGNMENT>>&);
}