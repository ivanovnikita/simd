#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>

namespace simd
{
    template <>
    struct vector_traits<float, avx_tag>
    {
        using value_type = float;
        using intr_type = __m256;
        static constexpr uint8_t required_alignment = 32;
    };

    template <>
    inline vector<float, avx_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    template <>
    inline void vector<float, avx_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm256_load_ps(ptr);
    }

    template <>
    inline vector<float, avx_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    template <>
    inline vector<float, avx_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    template <>
    inline void vector<float, avx_tag>::setzero_p() noexcept
    {
        m_values = _mm256_setzero_ps();
    }

    template <>
    inline void vector<float, avx_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm256_store_ps(ptr, m_values);
    }

    template <>
    inline vector<float, avx_tag> operator+(vector<float, avx_tag> lhs, vector<float, avx_tag> rhs) noexcept
    {
        return _mm256_add_ps(lhs, rhs);
    }

    template <>
    inline vector<float, avx_tag>& vector<float, avx_tag>::operator+=(vector<float, avx_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }
}
