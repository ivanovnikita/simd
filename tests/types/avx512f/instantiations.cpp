#include "../types.hpp"
#include "types/avx512f/vector.hpp"

using Avx512fVectors = testing::Types
<
    TypePair<simd::avx512f_tag, float>
    , TypePair<simd::avx512f_tag, double>
>;

INSTANTIATE_TYPED_TEST_CASE_P
(
    avx512f_type_test
    , simd_type_test
    , Avx512fVectors
);
