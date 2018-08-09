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
class accumulate : public testing::Test
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
    TypePair<simd::scalar_tag, float>
    , TypePair<simd::scalar_tag, double>
    , TypePair<simd::sse_tag, float>
    , TypePair<simd::avx_tag, float>
    , TypePair<simd::avx_tag, double>
>;

TYPED_TEST_CASE(accumulate, AccumulateTestingPairs);

TYPED_TEST(accumulate, equal_elements)
{
    using simd_tag = typename TypeParam::simd_tag;

    EXPECT_EQ(this->count, simd::detail::accumulate<simd_tag>(this->values));
}