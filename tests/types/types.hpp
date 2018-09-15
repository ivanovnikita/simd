#include "../tests_utils.h"

#include "static_aligned_allocator.hpp"
#include "types/vector.hpp"

#include <gtest/gtest.h>

template <class T>
class simd_type_test : public testing::Test
{
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

TYPED_TEST_P(simd_type_test, constr_by_ptr)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    simd::aligned_vector<value_type> init_values(capacity, value_type{});
    for (size_t i = 0; i < init_values.size(); ++i)
    {
        ++init_values[i];
    }

    const simd::vector<value_type, simd_tag> vector(init_values.data());

    simd::aligned_vector<value_type> values(capacity);
    vector.store_p(values.data());

    for (size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(init_values[i], values[i]);
    }
}

TYPED_TEST_P(simd_type_test, cast_to_instr)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    const intr_type default_value{1};

    const simd::vector<value_type, simd_tag> first(default_value);
    const simd::vector<value_type, simd_tag> second(static_cast<intr_type>(first));

    simd::aligned_vector<value_type> first_values(capacity);
    first.store_p(first_values.data());

    simd::aligned_vector<value_type> second_values(capacity);
    second.store_p(second_values.data());

    for (size_t i = 0; i < first_values.size(); ++i)
    {
        EXPECT_EQ(first_values[i], second_values[i]);
    }
}

TYPED_TEST_P(simd_type_test, setzero)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    const value_type zero{};
    simd::aligned_vector<value_type> init_values(capacity, zero);
    for (size_t i = 0; i < init_values.size(); ++i)
    {
        ++init_values[i];
        ASSERT_NE(zero, init_values[i]);
    }

    simd::vector<value_type, simd_tag> vector(init_values.data());
    vector.setzero_p();

    simd::aligned_vector<value_type> values(capacity);
    vector.store_p(values.data());

    for (size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(zero, values[i]);
    }
}

TYPED_TEST_P(simd_type_test, load_store)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    simd::aligned_vector<value_type> init_values(capacity, value_type{});
    for (size_t i = 0; i < init_values.size(); ++i)
    {
        ++init_values[i];
    }

    simd::vector<value_type, simd_tag> vector;
    vector.load_p(init_values.data());

    simd::aligned_vector<value_type> values(capacity);
    vector.store_p(values.data());

    for (size_t i = 0; i < values.size(); ++i)
    {
        EXPECT_EQ(init_values[i], values[i]);
    }
}

TYPED_TEST_P(simd_type_test, load_partial)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    for (size_t partial_size = 1; partial_size <= capacity; ++partial_size)
    {
        simd::aligned_vector<value_type> init_values(partial_size, value_type{});
        for (size_t i = 0; i < init_values.size(); ++i)
        {
            ++init_values[i];
        }

        simd::vector<value_type, simd_tag> vector;
        vector.load_partial(init_values.data(), partial_size);

        simd::aligned_vector<value_type> values(capacity);
        vector.store_p(values.data());

        for (size_t i = 0; i < init_values.size(); ++i)
        {
            EXPECT_EQ(init_values[i], values[i]);
        }

        const value_type zero{};
        for (size_t i = init_values.size(); i < values.size(); ++i)
        {
            EXPECT_EQ(zero, values[i]);
        }
    }
}

TYPED_TEST_P(simd_type_test, add_assign)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    simd::aligned_vector<value_type> first_values(capacity, value_type{});
    for (size_t i = 0; i < first_values.size(); ++i)
    {
        ++first_values[i];
    }

    simd::aligned_vector<value_type> second_values(capacity, value_type{});
    for (size_t i = 0; i < second_values.size(); ++i)
    {
        ++second_values[i];
        ++second_values[i];
    }

    simd::vector<value_type, simd_tag> first(first_values.data());
    const simd::vector<value_type, simd_tag> second(second_values.data());

    first += second;

    simd::aligned_vector<value_type> result(capacity);
    first.store_p(result.data());

    for (size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(first_values[i] + second_values[i], result[i]);
    }
}

TYPED_TEST_P(simd_type_test, add)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    simd::aligned_vector<value_type> first_values(capacity, value_type{});
    for (size_t i = 0; i < first_values.size(); ++i)
    {
        ++first_values[i];
    }

    simd::aligned_vector<value_type> second_values(capacity, value_type{});
    for (size_t i = 0; i < second_values.size(); ++i)
    {
        ++second_values[i];
        ++second_values[i];
    }

    const simd::vector<value_type, simd_tag> first(first_values.data());
    const simd::vector<value_type, simd_tag> second(second_values.data());

    const simd::vector<value_type, simd_tag> third = first + second;

    simd::aligned_vector<value_type> result(capacity);
    third.store_p(result.data());

    for (size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_EQ(first_values[i] + second_values[i], result[i]);
    }
}

TYPED_TEST_P(simd_type_test, horizontal_add)
{
    using simd_tag = typename TypeParam::simd_tag;
    SKIP_UNAVAILABLE_TEST(simd_tag)

    using namespace simd;
    using value_type = typename TypeParam::value_type;

    using intr_type = typename simd::vector<value_type, simd_tag>::intr_type;
    const uint8_t capacity = simd::vector<value_type, simd_tag>::capacity;

    for (size_t partial_size = 0; partial_size < capacity; ++partial_size)
    {
        value_type sum{};

        simd::aligned_vector<value_type> init_values(capacity, value_type{});
        for (size_t i = 0; i < partial_size; ++i)
        {
            ++init_values[i];
            sum += init_values[i];
        }

        const simd::vector<value_type, simd_tag> vector(init_values.data());
        const auto result = simd::horizontal_add(vector);

        EXPECT_EQ(sum, result);
    }
}

REGISTER_TYPED_TEST_CASE_P
(
    simd_type_test
    , constr_by_instr
    , constr_by_ptr
    , cast_to_instr
    , setzero
    , load_store
    , load_partial
    , add_assign
    , add
    , horizontal_add
);