#include "instruction_sets/sse/operations.h"
#include "instruction_sets/avx/operations.h"
#include "instruction_sets/avx-512/operations.h"

#include <cstddef>
#include <iostream>

int main()
{
    simd_sse::do_stuff();
    simd_avx::do_stuff();
//    simd_avx512::do_stuff();

    return 0;
}
