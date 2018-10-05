#include "accumulate.h"

#include "instrset_detect.h"

#include "scalar_implementation.hpp"
#include "sse/declaration.hpp"
#include "avx/declaration.hpp"
#include "avx2/declaration.hpp"
#include "avx512f/declaration.hpp"
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
                    std::printf("accumulate<avx512f_tag, %s> chosen\n", typeid(T).name());
                    return accumulate<avx512f_tag, T>;
                }
            }

            if constexpr (has_function_accumulate<avx2_tag, T>::value)
            {
                if (has_avx2())
                {
                    std::printf("accumulate<avx2_tag, %s> chosen\n", typeid(T).name());
                    return accumulate<avx2_tag, T>;
                }
            }

            if constexpr (has_function_accumulate<avx_tag, T>::value)
            {
                if (has_avx())
                {
                    std::printf("accumulate<avx_tag, %s> chosen\n", typeid(T).name());
                    return accumulate<avx_tag, T>;
                }
            }

            if constexpr (has_function_accumulate<sse_tag, T>::value)
            {
                if (has_sse())
                {
                    std::printf("accumulate<sse_tag, %s> chosen\n", typeid(T).name());
                    return accumulate<sse_tag, T>;
                }
            }

            std::printf("accumulate<scalar_tag, %s> chosen\n", typeid(T).name());
            return accumulate<scalar_tag, T>;
        }
    }

#if defined(__clang__)
    template <typename T>
    accumulate_t<T>* const best_available_accumulate = init<T>();

    template accumulate_t<float>* const best_available_accumulate<float>;
    template accumulate_t<double>* const best_available_accumulate<double>;
    template accumulate_t<int8_t>* const best_available_accumulate<int8_t>;
    template accumulate_t<int16_t>* const best_available_accumulate<int16_t>;
    template accumulate_t<int32_t>* const best_available_accumulate<int32_t>;
    template accumulate_t<int64_t>* const best_available_accumulate<int64_t>;
#elif defined(__GNUC__)
    template <> accumulate_t<float>* best_available_accumulate<float> = init<float>();
    template <> accumulate_t<double>* best_available_accumulate<double> = init<double>();
    template <> accumulate_t<int8_t>* best_available_accumulate<int8_t> = init<int8_t>();
    template <> accumulate_t<int16_t>* best_available_accumulate<int16_t> = init<int16_t>();
    template <> accumulate_t<int32_t>* best_available_accumulate<int32_t> = init<int32_t>();
    template <> accumulate_t<int64_t>* best_available_accumulate<int64_t> = init<int64_t>();
#endif
}