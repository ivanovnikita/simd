#include "accumulate/default/implementation.h"
#include "accumulate/sse/implementation.h"
#include "accumulate/avx/implementation.h"
#include "accumulate/avx512f/implementation.h"

#include <benchmark/benchmark.h>

class BM_accumulate : public benchmark::Fixture
{
public:
    void SetUp(benchmark::State& state) override
    {
        values.resize(state.range(0));
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = i % 2;
        }
    }

protected:
    std::vector<float, simd::static_aligned_allocator<float, 64>> values;
};

BENCHMARK_DEFINE_F(BM_accumulate, def)(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(simd::def::accumulate(values));
    }
}

BENCHMARK_DEFINE_F(BM_accumulate, sse)(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(simd::sse::accumulate(values));
    }
}

BENCHMARK_DEFINE_F(BM_accumulate, avx)(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(simd::avx::accumulate(values));
    }
}

//BENCHMARK_REGISTER_F(BM_accumulate, def)->RangeMultiplier(2)->Range(64, 2LL << 24);
//BENCHMARK_REGISTER_F(BM_accumulate, sse)->RangeMultiplier(2)->Range(64, 2LL << 24);
//BENCHMARK_REGISTER_F(BM_accumulate, avx)->RangeMultiplier(2)->Range(64, 2LL << 24);
BENCHMARK_REGISTER_F(BM_accumulate, def)->Arg(2LL << 15);
BENCHMARK_REGISTER_F(BM_accumulate, sse)->Arg(2LL << 15);
BENCHMARK_REGISTER_F(BM_accumulate, avx)->Arg(2LL << 15);

BENCHMARK_MAIN();
