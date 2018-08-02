#include "instruction_sets/none/operations.h"
#include "instruction_sets/sse/operations.h"
#include "instruction_sets/avx/operations.h"
#include "instruction_sets/avx-512/operations.h"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <iostream>
#include <chrono>

void BM_accumulate_base(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(simd_none::accumulate(0));
    }
}

void BM_accumulate_sse(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(simd_sse::accumulate(0));
    }
}

void BM_accumulate_avx(benchmark::State& state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(simd_avx::accumulate(0));
    }
}

BENCHMARK(BM_accumulate_base);
BENCHMARK(BM_accumulate_sse);
BENCHMARK(BM_accumulate_avx);

BENCHMARK_MAIN();
