#include "accumulate/sse/definition.hpp"
#include "types/sse/vector.hpp"

#include "../fixture.hpp"
#include "../manual_operations_traits.hpp"

BASELINE_F(AccumulateFloatUnroll, Sse_1, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<1>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_2, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<2>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_3, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<3>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_4, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<4>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_5, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<5>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_6, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<6>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_7, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<7>>(values));
}

BENCHMARK_F(AccumulateFloatUnroll, Sse_8, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag, float, simd::manual_operations_traits<8>>(values));
}