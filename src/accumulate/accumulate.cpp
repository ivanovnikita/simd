#include "accumulate.h"

#include "instrset_detect.h"

#include "scalar_implementation.hpp"
#include "sse/instantiation.h"
#include "avx/instantiation.h"
#include "avx512f/instantiation.h"
#include "simd_implementation_decl.hpp"
#include "types/simd_tags.h"

#include <cstdio>

namespace simd::detail
{
    namespace
    {
        template <typename, typename, typename = std::void_t<>>
        struct has_function_accumulate : std::false_type{};

        template <typename simd_tag, typename T>
        struct has_function_accumulate
        <
            simd_tag
            , T
            , std::void_t<decltype(accumulate<simd_tag>(aligned_vector<T>{}))>
        > : std::true_type{};

        static_assert(has_function_accumulate<sse_tag, float>::value);
        static_assert(not has_function_accumulate<sse_tag, double>::value);

        template <typename T>
        accumulate_t<T>* init()
        {
            if constexpr (has_function_accumulate<avx512f_tag, T>::value)
            {
                if (has_avx512f())
                {
                    std::printf("avx512f::accumulate<%s> chosen\n", typeid(T).name());
                    return accumulate<avx512f_tag, T>;
                }
            }

            if constexpr (has_function_accumulate<avx_tag, T>::value)
            {
                if (has_avx())
                {
                    std::printf("avx::accumulate<%s> chosen\n", typeid(T).name());
                    return accumulate<avx_tag, T>;
                }
            }

            if constexpr (has_function_accumulate<sse_tag, T>::value)
            {
                if (has_sse())
                {
                    std::printf("sse::accumulate<%s> chosen\n", typeid(T).name());
                    return accumulate<sse_tag, T>;
                }
            }

            std::printf("scalar::accumulate<%s> chosen\n", typeid(T).name());
            return accumulate<scalar_tag, T>;
        }
    }

    template <typename T>
    accumulate_t<T>* const best_available_accumulate = init<T>();

    template accumulate_t<float>* const best_available_accumulate<float>;
    template accumulate_t<double>* const best_available_accumulate<double>;
}