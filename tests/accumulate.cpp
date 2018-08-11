#include "accumulate/scalar_implementation.hpp"
#include "accumulate/sse/instantiation.h"
#include "accumulate/avx/instantiation.h"
#include "accumulate/avx2/instantiation.h"
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
        count = 111;
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
    , TypePair<simd::best_available_tag, int8_t>
    , TypePair<simd::scalar_tag, float>
    , TypePair<simd::scalar_tag, double>
    , TypePair<simd::sse_tag, float>
    , TypePair<simd::avx_tag, float>
    , TypePair<simd::avx_tag, double>
    , TypePair<simd::avx2_tag, int8_t>
>;

TYPED_TEST_CASE(accumulate_test, AccumulateTestingPairs);

TYPED_TEST(accumulate_test, equal_elements)
{
    using simd_tag = typename TypeParam::simd_tag;
    using namespace simd;
    using namespace simd::detail;

    EXPECT_EQ(this->count, accumulate<simd_tag>(this->values));
}

template <class T>
class accumulate_auto_test : public testing::Test
{
protected:
    using value_type = T;

    void SetUp() override
    {
        count = 111;
        values.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            values[i] = 1;
        }
    }

    simd::aligned_vector<value_type> values;
    size_t count;
};

using AccumulateAutoTesting = testing::Types
<
    float
    , double
    , int8_t
>;

TYPED_TEST_CASE(accumulate_auto_test, AccumulateAutoTesting);

TYPED_TEST(accumulate_auto_test, linkage)
{
    using namespace simd;

    EXPECT_EQ(this->count, accumulate(this->values));
}