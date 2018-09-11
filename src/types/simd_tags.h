#pragma once

namespace simd
{
    struct avx512f_tag {};
    struct avx2_tag {};
    struct avx_tag {};
    struct sse_tag {};
    struct sse2_tag {};
    struct scalar_tag {};
    struct best_available_tag {};
}
