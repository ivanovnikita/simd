#pragma once

#include "simd_implementation_decl.hpp"
#include "types/vector.hpp"

namespace simd::detail
{
    template <typename simd_tag, typename T>
    T accumulate_simd_impl(const aligned_vector<T>& values)
    {
        constexpr uint8_t step = vector<T, simd_tag>::capacity;

        vector<T, simd_tag> simd_result;
        simd_result.setzero_p();

        size_t i = step;
        for (; i < values.size(); i += step)
        {
            simd_result += vector<T, simd_tag>(&values[i - step]);
        }

        i -= step;
        T result = 0;
        for (; i < values.size(); ++i)
        {
            result += values[i];
        }

        alignas(vector<T, simd_tag>::required_alignment) T result_values[vector<T, simd_tag>::capacity];
        simd_result.store_p(result_values);
        for (const T r : result_values)
        {
            result += r;
        }

        return result;
    }
}
