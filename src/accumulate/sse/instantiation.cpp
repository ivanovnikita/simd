#include "definition.hpp"
#include "types/sse/vector.hpp"

namespace simd::detail
{
    template float accumulate<sse_tag, float>(const aligned_vector<float>&);
}