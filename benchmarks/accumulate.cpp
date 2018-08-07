#include "accumulate/scalar/instantiation.h"
#include "accumulate/sse/instantiation.h"
#include "accumulate/avx/instantiation.h"
#include "accumulate/avx512f/instantiation.h"

#include "accumulate/accumulate.h"

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
    celero::DoNotOptimizeAway(simd::sse::accumulate(values));
}

BENCHMARK_F(Accumulate, Avx, AccumulateFixture, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::avx::accumulate(values));
}

BENCHMARK_F(Accumulate, AutoChoosen, AccumulateFixture, 10, 10000)
{
    celero::DoNotOptimizeAway(simd::accumulate(values));
}

//BENCHMARK_F(Accumulate, Avx512, AccumulateFixture, 10, 100)
//{
//    celero::DoNotOptimizeAway(simd::avx512::accumulate(values));
//}
