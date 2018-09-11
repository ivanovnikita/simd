#include "types/sse2/vector.hpp"

#include "../load_partial_fixture.hpp"

BENCHMARK_F(LoadPartial_1, Sse2, LoadPartialFixture<float>, 100, 10000)
{
    simd::vector<float, simd::sse2_tag> vector;
    for (size_t i = 0; i < values.size(); i += simd::vector<float, simd::sse2_tag>::capacity)
    {
        vector.load_partial(values.data(), 1);
    }
}

BENCHMARK_F(LoadPartial_2, Sse2, LoadPartialFixture<float>, 100, 10000)
{
    simd::vector<float, simd::sse2_tag> vector;
    for (size_t i = 0; i < values.size(); i += simd::vector<float, simd::sse2_tag>::capacity)
    {
        vector.load_partial(values.data(), 2);
    }
}

BENCHMARK_F(LoadPartial_3, Sse2, LoadPartialFixture<float>, 100, 10000)
{
    simd::vector<float, simd::sse2_tag> vector;
    for (size_t i = 0; i < values.size(); i += simd::vector<float, simd::sse2_tag>::capacity)
    {
        vector.load_partial(values.data(), 3);
    }
}
