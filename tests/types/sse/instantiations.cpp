#include "../types.hpp"
#include "types/sse/vector.hpp"

using SseVectors = testing::Types
<
    TypePair<simd::sse_tag, float>
>;

INSTANTIATE_TYPED_TEST_CASE_P
(
    sse_type_test
    , simd_type_test
    , SseVectors
);
