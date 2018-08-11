#include "definition.hpp"
#include "types/avx2/vector.hpp"

namespace simd::detail
{
    template int8_t accumulate<avx2_tag, int8_t>(const aligned_vector<int8_t>&);
}