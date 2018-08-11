#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>

namespace simd
{
    template <>
    class vector<double, avx512f_tag>
    {
    public:
        using value_type = double;
        using intr_type = __m512d;
        static constexpr uint8_t capacity = 8;
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

    vector<double, avx512f_tag> operator+(vector<double, avx512f_tag>, vector<double, avx512f_tag>) noexcept;

    // definitions

    inline vector<double, avx512f_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    inline vector<double, avx512f_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    inline vector<double, avx512f_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    inline void vector<double, avx512f_tag>::setzero_p() noexcept
    {
        m_values = _mm512_setzero_pd();
    }

    inline void vector<double, avx512f_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm512_load_pd(ptr);
    }

    inline void vector<double, avx512f_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm512_store_pd(ptr, m_values);
    }

    inline vector<double, avx512f_tag>& vector<double, avx512f_tag>::operator+=(vector<double, avx512f_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    inline vector<double, avx512f_tag> operator+(vector<double, avx512f_tag> lhs, vector<double, avx512f_tag> rhs) noexcept
    {
        return _mm512_add_pd(lhs, rhs);
    }
}

