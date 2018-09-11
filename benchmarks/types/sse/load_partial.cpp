#include "types/sse/vector.hpp"

#include "../load_partial_fixture.hpp"

BASELINE_F(LoadPartial_1, Sse, LoadPartialFixture<float>, 100, 10000)
{
    simd::vector<float, simd::sse_tag> vector;
    for (size_t i = 0; i < values.size(); i += simd::vector<float, simd::sse_tag>::capacity)
    {
        vector.load_partial(values.data(), 1);
    }
}

BASELINE_F(LoadPartial_2, Sse, LoadPartialFixture<float>, 100, 10000)
{
    simd::vector<float, simd::sse_tag> vector;
    for (size_t i = 0; i < values.size(); i += simd::vector<float, simd::sse_tag>::capacity)
    {
        vector.load_partial(values.data(), 2);
    }
}

BASELINE_F(LoadPartial_3, Sse, LoadPartialFixture<float>, 100, 10000)
{
    simd::vector<float, simd::sse_tag> vector;
    for (size_t i = 0; i < values.size(); i += simd::vector<float, simd::sse_tag>::capacity)
    {
        vector.load_partial(values.data(), 3);
    }
}