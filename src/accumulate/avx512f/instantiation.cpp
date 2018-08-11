#include "definition.hpp"
#include "types/avx512f/vector.hpp"

namespace simd::detail
{
    template float accumulate<avx512f_tag, float>(const aligned_vector<float>&);
    template double accumulate<avx512f_tag, double>(const aligned_vector<double>&);
}