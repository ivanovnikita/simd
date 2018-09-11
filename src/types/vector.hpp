#pragma once

#include <cstdint>

namespace simd
{
    template <typename T, typename simd_tag>
    struct vector_traits;

    template <typename T, typename simd_tag>
    class vector
    {
    public:
        using value_type = typename vector_traits<T, simd_tag>::value_type;
        using intr_type = typename vector_traits<T, simd_tag>::intr_type;
        static constexpr uint8_t capacity = sizeof(intr_type) / sizeof(T);
        static constexpr uint8_t required_alignment = vector_traits<T, simd_tag>::required_alignment;

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

    template <typename T, typename simd_tag>
    vector<T, simd_tag> operator+(vector<T, simd_tag>, vector<T, simd_tag>) noexcept;
}
