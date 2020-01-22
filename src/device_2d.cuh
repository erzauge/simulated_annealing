#pragma once

#include "setup.cuh"
#include <curand_kernel.h>

texture<float,cudaTextureType1D, cudaReadModeElementType> &get_boltz(void);

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_J_xi(void);

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_J_yi(void);

__global__ void metrpolis_2d(spin_t *s_u,spin_t *s_i,curandStatePhilox4_32_10_t *random,float * boltz,int L, long J_offset);
__global__ void J_order(int2 *J_xi_d, int2 *J_yi_d,unsigned int *buffer, int L, long N);
__global__ void bittoint(spin_t *s, long *sM, int b);
__global__ void measure_M(spin_t *s, float *M_buf,long N);
__global__ void measure_EJ_2d(spin_t *s, float * EJ_buf, long N, int L);
__global__ void measure_EJ_M_2d(spin_t *s_1, float * EJ_buf, float * M_buf, long N, int L, long rand_offset);
__global__ void checkerbord_switch_2d(spin_t *s_1,spin_t *s_2, int L);
__global__ void swap_2d(spin_t *s_1,spin_t *s_2,spin_t mask,long N);
