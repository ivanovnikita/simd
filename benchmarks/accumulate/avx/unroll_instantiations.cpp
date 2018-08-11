#include "accumulate/avx/definition.hpp"
#include "types/avx/vector.hpp"

#include "../fixture.hpp"
#include "../manual_operations_traits.hpp"

BASELINE_F(AccumulateFloatUnroll, Avx_1, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<1>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_2, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<2>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_3, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<3>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_4, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<4>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_5, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<5>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_6, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<6>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_7, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<7>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Avx_8, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, float, simd::manual_operations_traits<8>>(values));
}


BASELINE_F(AccumulateDoubleUnroll, Avx_1, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<1>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_2, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<2>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_3, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<3>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_4, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<4>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_5, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<5>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_6, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<6>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_7, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<7>>(values));
}

BENCHMARK_F(AccumulateDoubleUnroll, Avx_8, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag, double, simd::manual_operations_traits<8>>(values));
}