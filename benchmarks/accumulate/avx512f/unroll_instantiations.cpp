#include "accumulate/avx512f/definition.hpp"
#include "types/avx512f/vector.hpp"

#include "../fixture.hpp"
#include "../manual_operations_traits.hpp"

//BASELINE_F(AccumulateFloatUnroll, Avx512f_1, AccumulateFixture<float>, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag, float, simd::manual_operations_traits<1>>(values));
//}
//
//BENCHMARK_F(AccumulateFloatUnroll, Avx512f_3, AccumulateFixture<float>, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag, float, simd::manual_operations_traits<3>>(values));
//}
//
//
//BASELINE_F(AccumulateDoubleUnroll, Avx512f_1, AccumulateFixture<double>, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag, double, simd::manual_operations_traits<1>>(values));
//}
//
//BENCHMARK_F(AccumulateDoubleUnroll, Avx512f_3, AccumulateFixture<double>, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag, double, simd::manual_operations_traits<3>>(values));
//}
