#include "../tests_utils.h"

#include "static_aligned_allocator.hpp"
#include "types/vector.hpp"

#include <gtest/gtest.h>

template <class T>
class simd_type_test : public testing::Test
{
protected:
    using value_type = typename T::value_type;

    void SetUp() override
    {

    }
};

TYPED_TEST_CASE_P(simd_type_test);

TYPED_TEST_P(simd_type_test, constr_by_instr)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    const auto zero = 0;
    const intr_type default_value{zero};

    const simd::vector<value_type, simd_tag> vector(default_value);

    const auto non_zero = static_cast<value_type>(1);
    simd::aligned_vector<value_type> values(capacity, non_zero);

    vector.store_p(values.data());

    for (size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(zero, values[i]);
    }
}

REGISTER_TYPED_TEST_CASE_P
(
    simd_type_test
    , constr_by_instr
);