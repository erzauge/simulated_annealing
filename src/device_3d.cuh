#pragma once

#include <curand_kernel.h>

#include "setup.cuh"

texture<float,cudaTextureType1D, cudaReadModeElementType> &get_3d_boltz(void);

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_3d_J_xi(void);

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_3d_J_yi(void);

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_3d_J_zi(void);

__global__ void metrpolis_3d(spin_t *s_u,spin_t *s_i,curandStatePhilox4_32_10_t *random,float *boltz,int L,long J_offset);
__global__ void J_order_3d(int2 *J_xi_d,  int2 *J_yi_d, int2 *J_zi_d, unsigned int *buffer, int L, long N);
__global__ void measure_EJ_M_3d(spin_t *s1, float * EJ_buf, float * M_buf, long N, int L);
__device__ __forceinline__ void updat_spin_split(spin_t &s_i,spin_t &s_w,spin_t &s_n,spin_t &s_u,spin_t &s_e,spin_t &s_s,spin_t &s_d,float &rand1,float &rand2,float *boltz,spin_t &Jw,spin_t &Jn,spin_t &Ju,spin_t &Je,spin_t &Js,spin_t &Jd);
__device__ __forceinline__ void updat_spin(spin_t &s_i,spin_t &s_w,spin_t &s_n,spin_t &s_u,spin_t &s_e,spin_t &s_s,spin_t &s_d,float &rand1,float *boltz,spin_t &Jw,spin_t &Jn,spin_t &Ju,spin_t &Je,spin_t &Js,spin_t &Jd);

__global__ void checkerbord_switch_3d(spin_t *s_1,spin_t *s_2, int L,int L2);
__global__ void swap_3d(spin_t *s_1,spin_t *s_2,spin_t mask,long N);
