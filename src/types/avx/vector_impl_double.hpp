#pragma once

#include "vector.hpp"

#include <immintrin.h>

#include <cstdint>

namespace simd::avx
{
    template <>
    class vector<double>
    {
    public:
        using value_type = double;
        using intr_type = __m256d;
        static constexpr uint8_t capacity = 4;
        static constexpr uint8_t required_alignment = 32;

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

    vector<double> operator+(vector<double>, vector<double>) noexcept;

    // definitions

    vector<double>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    vector<double>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    vector<double>::operator intr_type() const noexcept
    {
        return m_values;
    }

    void vector<double>::setzero_p() noexcept
    {
        m_values = _mm256_setzero_pd();
    }

    void vector<double>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm256_load_pd(ptr);
    }

    void vector<double>::store_p(value_type* ptr) const noexcept
    {
        _mm256_store_pd(ptr, m_values);
    }

    vector<double>& vector<double>::operator+=(vector<double> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    vector<double> operator+(vector<double> lhs, vector<double> rhs) noexcept
    {
        return _mm256_add_pd(lhs, rhs);
    }
}

