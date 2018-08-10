#pragma once

#include "simd_implementation_decl.hpp"
#include "types/vector.hpp"
#include "types/operations_traits.hpp"

#include <cassert>

namespace simd::detail
{
    template <typename simd_tag, typename T>
    T accumulate_simd_impl(const aligned_vector<T>& values)
    {
        const uint8_t adds_per_cycle = operations_traits::adds_per_cycle<simd_tag, T>;
//        const uint8_t adds_per_cycle = 3;
        assert(adds_per_cycle >= 1);

        std::vector<vector<T, simd_tag>> simd_results(adds_per_cycle);
        for (auto& simd_result : simd_results)
        {
            simd_result.setzero_p();
        }

        constexpr size_t substep = vector<T, simd_tag>::capacity;
        const size_t step = substep * simd_results.size();
        size_t i = step;
        for (; i < values.size(); i += step)
        {
            for (size_t j = 0; j < simd_results.size(); ++j)
            {
                simd_results[j] += vector<T, simd_tag>(&values[i - step + j * substep]);
            }
        }

        for (size_t i = 1; i < simd_results.size(); ++i)
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
