#include "accumulate/avx2/definition.hpp"
#include "types/avx2/vector.hpp"

#include "../fixture.hpp"
#include "../manual_operations_traits.hpp"

BASELINE_F(AccumulateInt8Unroll, Avx2_1, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<1>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_2, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<2>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_3, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<3>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_4, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<4>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_5, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<5>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_6, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<6>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_7, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<7>>(values));
}

BENCHMARK_F(AccumulateInt8Unroll, Avx2_8, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int8_t, simd::manual_operations_traits<8>>(values));
}

BASELINE_F(AccumulateInt16Unroll, Avx2_1, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<1>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_2, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<2>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_3, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<3>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_4, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<4>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_5, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<5>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_6, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<6>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_7, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<7>>(values));
}

BENCHMARK_F(AccumulateInt16Unroll, Avx2_8, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag, int16_t, simd::manual_operations_traits<8>>(values));
}