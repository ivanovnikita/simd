#include "instantiation.h"
#include "../simd_implementation_def.hpp"
#include "types/avx2/vector.hpp"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename T
        , typename std::enable_if_t
        <
            std::is_same_v<simd_tag, avx2_tag>
            and
            (
                std::is_same_v<T, int8_t>
            )
        >*
    >
    T accumulate(const aligned_vector<T>& values)
    {
        return accumulate_simd_impl<simd_tag>(values);
    }

    template int8_t accumulate<avx2_tag, int8_t, nullptr>(const aligned_vector<int8_t>&);
}