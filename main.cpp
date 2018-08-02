#include "instruction_sets/none/operations.h"
#include "instruction_sets/sse/operations.h"
#include "instruction_sets/avx/operations.h"
#include "instruction_sets/avx-512/operations.h"

#include <cstddef>
#include <iostream>
#include <chrono>

template <typename T>
void benchmark(T func)
{
    volatile float initial = 0;
    
    const auto start = std::chrono::high_resolution_clock::now();
    const auto result = func(initial);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout
        << "\tanswer: " << result
        << "\ttime: " << duration.count()
        << std::endl;
}

int main()
{
    std::cout << "none:\n";
    benchmark(simd_none::accumulate);

    std::cout << "sse:\n";
    benchmark(simd_sse::accumulate);

    std::cout << "avx:\n";
    benchmark(simd_avx::accumulate);

    return 0;
}
