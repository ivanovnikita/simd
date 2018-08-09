#include "accumulate/scalar_implementation.hpp"
#include "accumulate/sse/instantiation.h"
#include "accumulate/avx/instantiation.h"
#include "accumulate/avx512f/instantiation.h"
#include "accumulate/accumulate.h"

#include <gtest/gtest.h>

template<typename S, typename V>
struct TypePair
{
    using simd_tag = S;
    using value_type = V;
};

template <class T>
class accumulate_test : public testing::Test
{
protected:
    using value_type = typename T::value_type;

    void SetUp() override
    {
        count = 200;
        values.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            values[i] = 1;
        }
    }

    simd::aligned_vector<value_type> values;
    size_t count;
};

using AccumulateTestingPairs = testing::Types
<
    TypePair<simd::best_available_tag, float>
    , TypePair<simd::best_available_tag, double>
    , TypePair<simd::scalar_tag, float>
    , TypePair<simd::scalar_tag, double>
    , TypePair<simd::sse_tag, float>
    , TypePair<simd::avx_tag, float>
    , TypePair<simd::avx_tag, double>
>;

TYPED_TEST_CASE(accumulate_test, AccumulateTestingPairs);

TYPED_TEST(accumulate_test, equal_elements)
{
    using simd_tag = typename TypeParam::simd_tag;
    using namespace simd;
    using namespace simd::detail;

    EXPECT_EQ(this->count, accumulate<simd_tag>(this->values));
}