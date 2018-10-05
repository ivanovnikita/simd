#include "accumulate/scalar_implementation.hpp"
#include "accumulate/sse/declaration.hpp"
#include "accumulate/avx/declaration.hpp"
#include "accumulate/avx2/declaration.hpp"
#include "accumulate/avx512f/declaration.hpp"
#include "accumulate/accumulate.h"
#include "types/simd_tags.h"

#include "fixture.hpp"
#include "manual_operations_traits.hpp"

BASELINE_F(AccumulateFloat, Scalar, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::scalar_tag>(values));
}

BENCHMARK_F(AccumulateFloat, Sse, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag>(values));
}

BENCHMARK_F(AccumulateFloat, Avx, AccumulateFixture<float>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag>(values));
}

//BENCHMARK_F(Accumulate, Avx512f, AccumulateFixture<float>, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag>(values));
//}

BASELINE_F(AccumulateDouble, Scalar, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::scalar_tag>(values));
}

BENCHMARK_F(AccumulateDouble, Avx, AccumulateFixture<double>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag>(values));
}

//BENCHMARK_F(Accumulate, Avx512f, AccumulateFixture<float>, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag>(values));
//}

BASELINE_F(AccumulateInt8, Scalar, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::scalar_tag>(values));
}

BENCHMARK_F(AccumulateInt8, Avx2, AccumulateFixture<int8_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag>(values));
}

BASELINE_F(AccumulateInt16, Scalar, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::scalar_tag>(values));
}

BENCHMARK_F(AccumulateInt16, Avx2, AccumulateFixture<int16_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag>(values));
}

BASELINE_F(AccumulateInt32, Scalar, AccumulateFixture<int32_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::scalar_tag>(values));
}

BENCHMARK_F(AccumulateInt32, Avx2, AccumulateFixture<int32_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag>(values));
}

BASELINE_F(AccumulateInt64, Scalar, AccumulateFixture<int64_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::scalar_tag>(values));
}

BENCHMARK_F(AccumulateInt64, Avx2, AccumulateFixture<int64_t>, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx2_tag>(values));
}