#include "instantiation.h"
#include "../simd_implementation_def.hpp"
#include "types/sse/vector.hpp"

namespace simd::detail
{
    template
    <
        typename simd_tag
        , typename T
        , typename std::enable_if_t
        <
            std::is_same_v<simd_tag, sse_tag>
            and std::is_same_v<T, float>
        >*
    >
    T accumulate(const aligned_vector<T>& values)
    {
        return accumulate_simd_impl<simd_tag>(values);
    }

    template float accumulate<sse_tag, float, nullptr>(const aligned_vector<float>&);
}