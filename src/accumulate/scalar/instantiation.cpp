#include "instantiation.h"
#include "../implementation.hpp"
#include "types/scalar/vector.hpp"

namespace simd::scalar
{
    template <typename T>
    T accumulate(const std::vector<T, static_aligned_allocator<T, MAX_REQUIRED_ALIGNMENT>>& values)
    {
        T result = 0;
        for (const T r : values)
        {
            result += r;
        }

        return result;
    }

    template float accumulate(const std::vector<float, static_aligned_allocator<float, MAX_REQUIRED_ALIGNMENT>>&);
    template double accumulate(const std::vector<double, static_aligned_allocator<double, MAX_REQUIRED_ALIGNMENT>>&);
}