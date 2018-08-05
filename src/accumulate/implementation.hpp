#pragma once

#include "static_aligned_allocator.hpp"

#include <vector>

namespace simd
{
    template <template <typename> typename Vector, typename T>
    T accumulate(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>& values)
    {
        constexpr uint8_t step = Vector<T>::capacity;

        Vector<T> simd_result;
        simd_result.setzero_p();

        size_t i = step;
        for (; i < values.size(); i += step)
        {
            simd_result += Vector<T>(&values[i - step]);
        }

        i -= step;
        T result = 0;
        for (; i < values.size(); ++i)
        {
            result += values[i];
        }

        alignas(Vector<T>::required_alignment) T result_values[Vector<T>::capacity];
        simd_result.store_p(result_values);
        for (const T r : result_values)
        {
            result += r;
        }

        return result;
    }
}
