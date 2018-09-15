#include "../types.hpp"
#include "types/avx/vector.hpp"

using AvxVectors = testing::Types
<
    TypePair<simd::avx_tag, float>
    , TypePair<simd::avx_tag, double>
>;

INSTANTIATE_TYPED_TEST_CASE_P
(
    avx_type_test
    , simd_type_test
    , AvxVectors
);
