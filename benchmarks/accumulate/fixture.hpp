#include "static_aligned_allocator.hpp"

#include <celero/Celero.h>

template <typename T>
class AccumulateFixture : public celero::TestFixture
{
public:
    std::vector<celero::TestFixture::ExperimentValue> getExperimentValues() const override
    {
        const int total_number_of_tests = 1;
        std::vector<celero::TestFixture::ExperimentValue> problem_space;
        problem_space.reserve(total_number_of_tests);

        int values_count = 8192;
//        while (values_count < (2 << total_number_of_tests))
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

    simd::aligned_vector<T> values;
};
