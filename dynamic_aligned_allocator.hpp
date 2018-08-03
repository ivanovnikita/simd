#pragma once

#include <xmmintrin.h>

#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace simd
{
    template <typename T>
    class dynamic_aligned_allocator
    {
    public:
        using value_type = T;
        using pointer = T*;
        using size_type = size_t;

        template<typename U>
        struct rebind
        {
            using other = dynamic_aligned_allocator<U>;
        };

        explicit dynamic_aligned_allocator(uint8_t alignment) noexcept;

        pointer allocate(size_type);
        void deallocate(pointer, size_type) noexcept;

    private:
        uint8_t m_alignment;
    };

    template <typename T>
    dynamic_aligned_allocator<T>::dynamic_aligned_allocator(uint8_t alignment) noexcept
        : m_alignment(alignment)
    {
    }

    template <typename T>
    typename dynamic_aligned_allocator<T>::pointer dynamic_aligned_allocator<T>::allocate(size_type n)
    {
        auto ptr = reinterpret_cast<pointer>(_mm_malloc(sizeof(T) * n, m_alignment));
        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    template <typename T>
    void dynamic_aligned_allocator<T>::deallocate(pointer ptr, size_type) noexcept
    {
        _mm_free(ptr);
    }
}