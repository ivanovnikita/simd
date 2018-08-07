#include "accumulate.h"

#include "instrset_detect.h"

#include "scalar/instantiation.h"
#include "sse/instantiation.h"
#include "avx/instantiation.h"
#include "avx512f/instantiation.h"

#include <cstdio>

namespace simd::detail
{
    namespace
    {
        template <typename T>
        accumulate_t<T>* init();

        template <>
        accumulate_t<float>* init()
        {
            if (has_avx512f()) { std::printf("avx512f::accumulate<float> choosen\n"); return avx512::accumulate<float>; }
            else if (has_avx()) { std::printf("avx::accumulate<float> choosen\n"); return avx::accumulate<float>; }
            else if (has_sse()) { std::printf("sse::accumulate<float> choosen\n"); return sse::accumulate<float>; }
            else { std::printf("scalar::accumulate<float> choosen\n"); return scalar::accumulate<float>; }
        }

        template <>
        accumulate_t<double>* init()
        {
            if (has_avx512f()) { std::printf("avx512f::accumulate<double> choosen\n"); return avx512::accumulate<double>; }
            else if (has_avx()) { std::printf("avx::accumulate<double> choosen\n"); return avx::accumulate<double>; }
            else { std::printf("scalar::accumulate<double> choosen\n"); return scalar::accumulate<double>; }
        }
    }

    template <typename T>
    accumulate_t<T>* const accumulate = init<T>();

    template accumulate_t<float>* const accumulate<float>;
    template accumulate_t<double>* const accumulate<double>;
}