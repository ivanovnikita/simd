#include "instantiation.h"
#include "../simd_implementation_def.hpp"
#include "types/avx512f/vector.hpp"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename T
        , typename std::enable_if_t
        <
            std::is_same_v<simd_tag, avx512f_tag>
            and
            (
                std::is_same_v<T, float>
                or std::is_same_v<T, double>
            )
        >*
    >
    T accumulate(const aligned_vector<T>& values)
    {
        return accumulate_simd_impl<simd_tag>(values);
    }

    template float accumulate<avx512f_tag, float, nullptr>(const aligned_vector<float>&);
    template double accumulate<avx512f_tag, double, nullptr>(const aligned_vector<double>&);
}