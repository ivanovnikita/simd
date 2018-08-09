#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>

namespace simd
{
    template <>
    class vector<float, avx512f_tag>
    {
    public:
        using value_type = float;
        using intr_type = __m512;
        static constexpr uint8_t capacity = 16;
        static constexpr uint8_t required_alignment = 64;

        vector() = default;
        vector(intr_type) noexcept;
        vector(const value_type*) noexcept;

        operator intr_type() const noexcept;

        void setzero_p() noexcept;

        void load_p(const value_type*) noexcept;
        void store_p(value_type*) const noexcept;

        vector& operator+=(vector) noexcept;

    private:
        intr_type m_values;
    };

    vector<float, avx512f_tag> operator+(vector<float, avx512f_tag>, vector<float, avx512f_tag>) noexcept;

    // definitions

    vector<float, avx512f_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    vector<float, avx512f_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    vector<float, avx512f_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    void vector<float, avx512f_tag>::setzero_p() noexcept
    {
        m_values = _mm512_setzero_ps();
    }

    void vector<float, avx512f_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm512_load_ps(ptr);
    }

    void vector<float, avx512f_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm512_store_ps(ptr, m_values);
    }

    vector<float, avx512f_tag>& vector<float, avx512f_tag>::operator+=(vector<float, avx512f_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    vector<float, avx512f_tag> operator+(vector<float, avx512f_tag> lhs, vector<float, avx512f_tag> rhs) noexcept
    {
        return _mm512_add_ps(lhs, rhs);
    }
}
