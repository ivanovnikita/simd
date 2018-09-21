#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cassert>

namespace simd
{
    template <>
    struct vector_traits<int32_t, avx2_tag>
    {
        using value_type = int32_t;
        using intr_type = __m256i;
        static constexpr uint8_t required_alignment = 32;
    };

    template <>
    inline vector<int32_t, avx2_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    template <>
    inline void vector<int32_t, avx2_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm256_load_si256(reinterpret_cast<const intr_type*>(ptr));
    }

    template <>
    inline vector<int32_t, avx2_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    template <>
    inline vector<int32_t, avx2_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    template <>
    inline void vector<int32_t, avx2_tag>::setzero_p() noexcept
    {
        m_values = _mm256_setzero_si256();
    }

    template <>
    inline void vector<int32_t , avx2_tag>::load_partial(const value_type* ptr, uint8_t n) noexcept
    {
        assert(n <= capacity);

        alignas(required_alignment) value_type values[capacity];
        assert(alignof(values) == required_alignment);

        value_type* values_ptr = &values[0];

        size_t i = 0;
        for (; i < n; ++i, ++ptr, ++values_ptr)
        {
            *values_ptr = *ptr;
        }
        for (; i < capacity;  ++i, ++values_ptr)
        {
            *values_ptr = 0;
        }
        load_p(&values[0]);
    }

    template <>
    inline void vector<int32_t, avx2_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm256_store_si256(reinterpret_cast<intr_type*>(ptr), m_values);
    }

    template <>
    inline vector<int32_t, avx2_tag> operator+(vector<int32_t, avx2_tag> lhs, vector<int32_t, avx2_tag> rhs) noexcept
    {
        return _mm256_add_epi32(lhs, rhs);
    }
    template <>
    inline vector<int32_t, avx2_tag>& vector<int16_t, avx2_tag>::operator+=(vector<int32_t, avx2_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    template <>
    inline int32_t horizontal_add(vector<int32_t, avx2_tag> v) noexcept
    {
        // v: [v8, ..., v1]
        // x - no matter

        // sum1: [x, x, v8+v7, v6+v5, x, x, v4+v3, v2+v1]
        __m256i sum1 = _mm256_hadd_epi32(v, v);

        // sum2: [x, x, x, v8:v5, x, x, x, v4:v1]
        __m256i sum2 = _mm256_hadd_epi32(sum1, sum1);

        // sum3: [x, x, x, v8:v5]
        __m128i sum3 = _mm256_extracti128_si256(sum2, 1);

        __m128i sum4 = _mm_add_epi32
        (
            _mm256_castsi256_si128(sum2) // [x, x, x, v4:v1]
            , sum3
        ); // sum4: [x, ..., v8:v1]

        return _mm_cvtsi128_si32(sum4); // [v8:v1]
    }
}
