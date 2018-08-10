#include "operations_traits.hpp"

#include <type_traits>

namespace simd
{
    namespace
    {
        struct default_cpu_tag {};

        struct default_operations_traits
        {
            template <typename cpu_tag, typename simd_tag, typename value_type>
            static const uint8_t adds_per_cycle;
        };

        // Intel Broadwell / Haswell / Ivy Bridge
        template<> const uint8_t default_operations_traits::adds_per_cycle<default_cpu_tag, sse_tag, float> = 3;
        template<> const uint8_t default_operations_traits::adds_per_cycle<default_cpu_tag, avx_tag, float> = 3;
        template<> const uint8_t default_operations_traits::adds_per_cycle<default_cpu_tag, avx_tag, double> = 3;
        template<> const uint8_t default_operations_traits::adds_per_cycle<default_cpu_tag, avx2_tag, int8_t> = 2;
        template<> const uint8_t default_operations_traits::adds_per_cycle<default_cpu_tag, avx512f_tag, float> = 3; // ?
        template<> const uint8_t default_operations_traits::adds_per_cycle<default_cpu_tag, avx512f_tag, double> = 3; // ?

        template <typename simd_tag, typename value_type>
        uint8_t init_adds_per_cycle()
        {
            // TODO: get cpu/architecture type at runtime and select values
            return default_operations_traits::adds_per_cycle<default_cpu_tag, simd_tag, value_type>;
        };
    }

    template <typename simd_tag, typename value_type>
    const uint8_t operations_traits::adds_per_cycle = init_adds_per_cycle<simd_tag, value_type>();

    // why there is a linkage error?
//    extern template const uint8_t operations_traits::adds_per_cycle<sse_tag, float>;
//    extern template const uint8_t operations_traits::adds_per_cycle<avx_tag, float>;
//    extern template const uint8_t operations_traits::adds_per_cycle<avx_tag, double>;
//    extern template const uint8_t operations_traits::adds_per_cycle<avx2_tag, int8_t>;
//    extern template const uint8_t operations_traits::adds_per_cycle<avx512f_tag, float>;
//    extern template const uint8_t operations_traits::adds_per_cycle<avx512f_tag, double>;

    template<> const uint8_t operations_traits::adds_per_cycle<sse_tag, float> = init_adds_per_cycle<sse_tag, float>();
    template<> const uint8_t operations_traits::adds_per_cycle<avx_tag, float> = init_adds_per_cycle<avx_tag, float>();
    template<> const uint8_t operations_traits::adds_per_cycle<avx_tag, double> = init_adds_per_cycle<avx_tag, double>();
    template<> const uint8_t operations_traits::adds_per_cycle<avx2_tag, int8_t> = init_adds_per_cycle<avx2_tag, int8_t>();
    template<> const uint8_t operations_traits::adds_per_cycle<avx512f_tag, float> = init_adds_per_cycle<avx512f_tag, float>();
    template<> const uint8_t operations_traits::adds_per_cycle<avx512f_tag, double> = init_adds_per_cycle<avx512f_tag, double>();
}