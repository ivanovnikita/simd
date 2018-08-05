#pragma once

#include "vector.hpp"

#include <xmmintrin.h>

#include <cstdint>

namespace simd::sse
{
    template <>
    class vector<float>
    {
    public:
        using value_type = float;
        using intr_type = __m128;
        static constexpr uint8_t capacity = 4;
        static constexpr uint8_t required_alignment = 16;

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

    vector<float> operator+(vector<float>, vector<float>) noexcept;

    // definitions

    vector<float>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    vector<float>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    vector<float>::operator intr_type() const noexcept
    {
        return m_values;
    }

    void vector<float>::setzero_p() noexcept
    {
        m_values = _mm_setzero_ps();
    }

    void vector<float>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm_load_ps(ptr);
    }

    void vector<float>::store_p(value_type* ptr) const noexcept
    {
        _mm_store_ps(ptr, m_values);
    }

    vector<float>& vector<float>::operator+=(vector<float> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    vector<float> operator+(vector<float> lhs, vector<float> rhs) noexcept
    {
        return _mm_add_ps(lhs, rhs);
    }
}
