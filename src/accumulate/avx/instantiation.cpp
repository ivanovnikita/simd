#include "definition.hpp"
#include "types/avx/vector.hpp"

namespace simd::detail
{
    template float accumulate<avx_tag, float>(const aligned_vector<float>&);
    template double accumulate<avx_tag, double>(const aligned_vector<double>&);
}