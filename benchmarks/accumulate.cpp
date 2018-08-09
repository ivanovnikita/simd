#include "accumulate/scalar_implementation.hpp"
#include "accumulate/sse/instantiation.h"
#include "accumulate/avx/instantiation.h"
#include "accumulate/avx512f/instantiation.h"
#include "accumulate/accumulate.h"
#include "types/simd_tags.h"

#include "instrset_detect.h"

#include <celero/Celero.h>

class AccumulateFixture : public celero::TestFixture
{
public:
    std::vector<celero::TestFixture::ExperimentValue> getExperimentValues() const override
    {
        const int total_number_of_tests = 13;
        std::vector<celero::TestFixture::ExperimentValue> problem_space;
        problem_space.reserve(total_number_of_tests);

        int values_count = 64;
        while (values_count < (2 << total_number_of_tests))
        {
            problem_space.emplace_back(static_cast<int64_t>(values_count));
            values_count *= 2;
        }

        return problem_space;
    }

    void setUp(const celero::TestFixture::ExperimentValue& experimentValue) override
    {
        values.resize(experimentValue.Value);
        for (int i = 0; i < this->values.size(); ++i)
        {
            values[i] = rand();
        }
    }

    void tearDown() override
    {
        values.clear();
    }

    std::vector<float, simd::static_aligned_allocator<float, simd::MAX_REQUIRED_ALIGNMENT>> values;
};

BASELINE_F(Accumulate, Scalar, AccumulateFixture, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::scalar::accumulate(values));
}

BENCHMARK_F(Accumulate, Sse, AccumulateFixture, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::sse_tag>(values));
}

BENCHMARK_F(Accumulate, Avx, AccumulateFixture, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx_tag>(values));
}

BENCHMARK_F(Accumulate, AutoChoosen, AccumulateFixture, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::accumulate(values));
}

//BENCHMARK_F(Accumulate, Avx512f, AccumulateFixture, 10, 10000)
//{
//    celero::DoNotOptimizeAway(simd::detail::accumulate<simd::avx512f_tag>(values));
//}
