#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>

namespace simd
{
    template <>
    struct vector_traits<double, avx512f_tag>
    {
        using value_type = double;
        using intr_type = __m512d ;
        static constexpr uint8_t required_alignment = 64;
    };

    template <>
    inline vector<double, avx512f_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    template <>
    inline void vector<double, avx512f_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm512_load_pd(ptr);
    }

    template <>
    inline vector<double, avx512f_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    template <>
    inline vector<double, avx512f_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    template <>
    inline void vector<double, avx512f_tag>::setzero_p() noexcept
    {
        m_values = _mm512_setzero_pd();
    }

    template <>
    inline void vector<double, avx512f_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm512_store_pd(ptr, m_values);
    }

    template <>
    inline vector<double, avx512f_tag> operator+(vector<double, avx512f_tag> lhs, vector<double, avx512f_tag> rhs) noexcept
    {
        return _mm512_add_pd(lhs, rhs);
    }

    template <>
    inline vector<double, avx512f_tag>& vector<double, avx512f_tag>::operator+=(vector<double, avx512f_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }
}

