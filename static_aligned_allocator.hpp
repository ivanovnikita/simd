#pragma once

#include <xmmintrin.h>

#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace simd
{
    template <typename T, int N>
    class static_aligned_allocator
    {
    public:
        using value_type = T;
        using pointer = T*;
        using size_type = size_t;

        template<typename U>
        struct rebind
        {
            using other = static_aligned_allocator<U, N>;
        };

        pointer allocate(size_type);
        void deallocate(pointer, size_type) noexcept;
    };

    template <typename T, int N>
    typename static_aligned_allocator<T, N>::pointer static_aligned_allocator<T, N>::allocate(size_type n)
    {
        auto ptr = reinterpret_cast<pointer>(_mm_malloc(sizeof(T) * n, N));
        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    template <typename T, int N>
    void static_aligned_allocator<T, N>::deallocate(pointer ptr, size_type) noexcept
    {
        _mm_free(ptr);
    }
}