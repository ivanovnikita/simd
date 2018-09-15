#pragma once

#include "vector.hpp"
#include "types/vector.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cassert>

namespace simd
{
    template <>
    struct vector_traits<double, avx_tag>
    {
        using value_type = double;
        using intr_type = __m256d;
        static constexpr uint8_t required_alignment = 32;
    };

    template <>
    inline vector<double, avx_tag>::vector(intr_type values) noexcept
        : m_values(values)
    {
    }

    template <>
    inline void vector<double, avx_tag>::load_p(const value_type* ptr) noexcept
    {
        m_values = _mm256_load_pd(ptr);
    }

    template <>
    inline vector<double, avx_tag>::vector(const value_type* ptr) noexcept
    {
        load_p(ptr);
    }

    template <>
    inline vector<double, avx_tag>::operator intr_type() const noexcept
    {
        return m_values;
    }

    template <>
    inline void vector<double, avx_tag>::setzero_p() noexcept
    {
        m_values = _mm256_setzero_pd();
    }

    template <>
    inline void vector<double, avx_tag>::load_partial(const value_type* ptr, uint8_t n) noexcept
    {
        assert(n <= capacity);

        switch (n)
        {
            case 1:
            {
                m_values = _mm256_setr_m128d(_mm_load_sd(ptr), _mm_setzero_pd());
                break;
            }
            case 2:
            {
                m_values = _mm256_setr_m128d(_mm_load_pd(ptr), _mm_setzero_pd());
                break;
            }
            case 3:
            {
                m_values = _mm256_setr_m128d(_mm_load_pd(ptr), _mm_load_sd(ptr));
                break;
            }
            case 4:
            {
                load_p(ptr);
                break;
            }
            default:
            {
                setzero_p();
                break;
            }
        }
    }

    template <>
    inline void vector<double, avx_tag>::store_p(value_type* ptr) const noexcept
    {
        _mm256_store_pd(ptr, m_values);
    }

    template <>
    inline vector<double, avx_tag> operator+(vector<double, avx_tag> lhs, vector<double, avx_tag> rhs) noexcept
    {
        return _mm256_add_pd(lhs, rhs);
    }

    template <>
    inline vector<double, avx_tag>& vector<double, avx_tag>::operator+=(vector<double, avx_tag> rhs) noexcept
    {
        m_values = *this + rhs;
        return *this;
    }

    template <>
    inline double horizontal_add(vector<double, avx_tag> v) noexcept
    {
        // v: [v4, v3, v2, v1]
        // x - no matter

        // t1: [x, v4 + v3, x, v2 + v1]
        __m256d t1 = _mm256_hadd_pd(v, v);

        // t2: [x, v4 + v3]
        __m128d t2 = _mm256_extractf128_pd(t1, 1);

        __m128d t3 = _mm_add_sd
        (
            _mm256_castpd256_pd128(t1) // [x, v2 + v1]
            , t2
        ); // t3: [x, x, x, v2 + v1 + v4 + v3]

        return _mm_cvtsd_f64(t3); // v2 + v1 + v4 + v3
    }
}

