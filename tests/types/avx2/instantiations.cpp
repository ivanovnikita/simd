#include "../types.hpp"
#include "types/avx2/vector.hpp"

using Avx2Vectors = testing::Types
<
    TypePair<simd::avx2_tag, int8_t >
    , TypePair<simd::avx2_tag, int16_t>
    , TypePair<simd::avx2_tag, int32_t>
>;

INSTANTIATE_TYPED_TEST_CASE_P
(
    avx2_type_test
    , simd_type_test
    , Avx2Vectors
);
