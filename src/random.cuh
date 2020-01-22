#pragma once
#include <curand_kernel.h>

__global__ void setup_radome(curandStateXORWOW_t *state, unsigned long long seed, unsigned long long N);
__global__ void setup_radome(curandStatePhilox4_32_10_t *state, unsigned long long seed, unsigned long long N);
__global__ void generate_kernel(curandStatePhilox4_32_10_t *random,unsigned int *buffer,unsigned long long N, int t);
