#pragma once

namespace simd
{
    bool has_sse() noexcept;
    bool has_sse2() noexcept;
    bool has_sse3() noexcept;
    bool has_ssse3() noexcept;
    bool has_sse4_1() noexcept;
    bool has_sse4_2() noexcept;
    bool has_avx() noexcept;
    bool has_avx2() noexcept;
    bool has_avx512f() noexcept;
    bool has_avx512vl() noexcept;
    bool has_avx512bw_dq() noexcept;
    bool has_avx512er() noexcept;
    bool has_fma3() noexcept;
    bool has_fma4() noexcept;
    bool has_xop() noexcept;
    bool has_f16c() noexcept;
}