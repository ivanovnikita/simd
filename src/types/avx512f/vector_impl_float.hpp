#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cassert>

namespace simd
{
    template <>
    struct vector_traits<float, avx512f_tag>
    {
        using value_type = float;
        using intr_type = __m512;
        static constexpr uint8_t required_alignment = 64;
    };

    template <>
    inline vector<float, avx512f_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    template <>
    inline void vector<float, avx512f_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm512_load_ps(ptr);
    }

    template <>
    inline vector<float, avx512f_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    template <>
    inline vector<float, avx512f_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    template <>
    inline void vector<float, avx512f_tag>::setzero_p() noexcept
    {
        m_values = _mm512_setzero_ps();
    }

    template <>
    inline void vector<float, avx512f_tag>::load_partial(const value_type* ptr, uint8_t n) noexcept
    {
        assert(n <= capacity);

        m_values = _mm512_maskz_load_ps(__mmask16((1 << n) - 1), ptr);
    }

    template <>
    inline void vector<float, avx512f_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm512_store_ps(ptr, m_values);
    }

    template <>
    inline vector<float, avx512f_tag> operator+(vector<float, avx512f_tag> lhs, vector<float, avx512f_tag> rhs) noexcept
    {
        return _mm512_add_ps(lhs, rhs);
    }

    template <>
    inline vector<float, avx512f_tag>& vector<float, avx512f_tag>::operator+=(vector<float, avx512f_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }
}
