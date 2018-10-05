#include "definition.hpp"
#include "types/avx2/vector.hpp"

namespace simd::detail
{
    template int8_t accumulate<avx2_tag, int8_t>(const aligned_vector<int8_t>&);
    template int16_t accumulate<avx2_tag, int16_t>(const aligned_vector<int16_t>&);
    template int32_t accumulate<avx2_tag, int32_t>(const aligned_vector<int32_t>&);
    template int64_t accumulate<avx2_tag, int64_t>(const aligned_vector<int64_t>&);
}