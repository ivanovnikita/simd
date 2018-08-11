#pragma once

#include "simd_implementation_decl.hpp"
#include "types/vector.hpp"

#include <cassert>

namespace simd::detail
{
    template <typename simd_tag, typename T, typename op_traits>
    T accumulate_simd_impl(const aligned_vector<T>& values)
    {
        const uint8_t adds_per_cycle = op_traits::adds_per_cycle;
        assert(adds_per_cycle >= 1);

        vector<T, simd_tag> simd_results[adds_per_cycle];
        for (size_t i = 0; i < adds_per_cycle; ++i)
        {
            simd_results[i].setzero_p();
        }

        constexpr size_t substep = vector<T, simd_tag>::capacity;
        const size_t step = substep * adds_per_cycle;
        size_t i = step;
        for (; i < values.size(); i += step)
        {
            const T* values_ptr = &values[i - step];
            for (size_t j = 0; j < adds_per_cycle; ++j, values_ptr += substep)
            {
                simd_results[j] += vector<T, simd_tag>(values_ptr);
            }
        }

        for (size_t i = 1; i < adds_per_cycle; ++i)
        {
            simd_results[0] += simd_results[i];
        }

        i -= step;
        T result = 0;
        for (; i < values.size(); ++i)
        {
            result += values[i];
        }

        alignas(vector<T, simd_tag>::required_alignment) T result_values[vector<T, simd_tag>::capacity];
        simd_results[0].store_p(result_values);
        for (const T r : result_values)
        {
            result += r;
        }

        return result;
    }
}
