#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <xmmintrin.h>

#include <cstdint>
#include <cassert>

namespace simd
{
    template <>
    class vector<float, sse_tag>
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
        void load_partial(const value_type*, uint8_t n) noexcept;
        void store_p(value_type*) const noexcept;

        vector& operator+=(vector) noexcept;

    private:
        intr_type m_values;
    };

    vector<float, sse_tag> operator+(vector<float, sse_tag>, vector<float, sse_tag>) noexcept;

    // definitions

    inline vector<float, sse_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    inline vector<float, sse_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    inline vector<float, sse_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    inline void vector<float, sse_tag>::setzero_p() noexcept
    {
        m_values = _mm_setzero_ps();
    }

    inline void vector<float, sse_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm_load_ps(ptr);
    }

    inline void vector<float, sse_tag>::load_partial(const value_type* ptr, uint8_t n) noexcept
    {
        assert(n <= capacity);

        switch (n)
        {
        case 1:
        {
            m_values = _mm_load_ss(ptr);
            break;
        }
        case 2:
        {
            m_values = _mm_setr_ps(*ptr, *(ptr + 1), 0.0f, 0.0f);
            break;
        }
        case 3:
        {
            m_values = _mm_setr_ps(*ptr, *(ptr + 1), *(ptr + 2), 0.0f);
            break;
        }
        case 4:
        {
            load_p(ptr);
        }
        default:
        {
            setzero_p();
        }
        }
    }

    inline void vector<float, sse_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm_store_ps(ptr, m_values);
    }

    inline vector<float, sse_tag>& vector<float, sse_tag>::operator+=(vector<float, sse_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    inline vector<float, sse_tag> operator+(vector<float, sse_tag> lhs, vector<float, sse_tag> rhs) noexcept
    {
        return _mm_add_ps(lhs, rhs);
    }
}
