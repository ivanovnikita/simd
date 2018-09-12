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
        const T* values_ptr = nullptr;
        for (; i < values.size(); i += step)
        {
            values_ptr = &values[i - step];
            for (size_t j = 0; j < adds_per_cycle; ++j, values_ptr += substep)
            {
                simd_results[j] += vector<T, simd_tag>(values_ptr);
            }
        }

        for (size_t r = 1; r < adds_per_cycle; ++r)
        {
            simd_results[0] += simd_results[r];
        }

        i = i - step + substep;
        values_ptr = &values[i - substep];
        for (; i < values.size(); i += substep, values_ptr += substep)
        {
            simd_results[0] += vector<T, simd_tag>(values_ptr);
        }

        i -= substep;
        vector<T, simd_tag> remaining;
        remaining.load_partial(&values[i], static_cast<uint8_t>(values.size() - i));
        simd_results[0] += remaining;

        T result = 0;
        result += horizontal_add(simd_results[0]);

        return result;
    }
}
