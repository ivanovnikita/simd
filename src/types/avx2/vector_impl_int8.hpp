#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>

namespace simd
{
    template <>
    class vector<int8_t, avx2_tag>
    {
    public:
        using value_type = int8_t;
        using intr_type = __m256i;
        static constexpr uint8_t capacity = 32;
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

    vector<int8_t, avx2_tag> operator+(vector<int8_t, avx2_tag>, vector<int8_t, avx2_tag>) noexcept;

    // definitions

    inline vector<int8_t, avx2_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    inline vector<int8_t, avx2_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    inline vector<int8_t, avx2_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    inline void vector<int8_t, avx2_tag>::setzero_p() noexcept
    {
        m_values = _mm256_setzero_si256();
    }

    inline void vector<int8_t, avx2_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm256_load_si256(reinterpret_cast<const intr_type*>(ptr));
    }

    inline void vector<int8_t, avx2_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm256_store_si256(reinterpret_cast<intr_type*>(ptr), m_values);
    }

    inline vector<int8_t, avx2_tag>& vector<int8_t, avx2_tag>::operator+=(vector<int8_t, avx2_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    inline vector<int8_t, avx2_tag> operator+(vector<int8_t, avx2_tag> lhs, vector<int8_t, avx2_tag> rhs) noexcept
    {
        return _mm256_add_epi8(lhs, rhs);
    }
}