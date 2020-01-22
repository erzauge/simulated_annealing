#include "device.cuh"
#include "device_3d.cuh"
#include "setup.cuh"


#include <stdio.h>

texture<float, cudaTextureType1D, cudaReadModeElementType> boltz;

texture<int2, cudaTextureType1D, cudaReadModeElementType> J_xi;

texture<int2, cudaTextureType1D, cudaReadModeElementType> J_yi;

texture<int2, cudaTextureType1D, cudaReadModeElementType> J_zi;



texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_3d_J_xi(void) {
  return J_xi;
}

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_3d_J_yi(void) {
  return J_yi;
}

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_3d_J_zi(void) {
  return J_zi;
}

__forceinline__ __device__ long xyz_id(int x, int y, int z, int L, int L2) {
  return z * L2 + y * L + x;
}

__global__ void metrpolis_3d(spin_t *s_u, spin_t *s_i,
                             curandStatePhilox4_32_10_t *random, float *boltz,
                             int L, long J_offset) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x < L && y < L && z < L) {
    long L2 = L * L;
    curandStatePhilox4_32_10_t rng = random[xyz_id(x, y, z, L, L2)];
    float rand1 = curand_uniform(&rng);
    #ifdef SPLIT_H
    float rand2 = curand_uniform(&rng);
    #endif
    random[xyz_id(x, y, z, L, L2)] = rng;
    spin_t Jw = int2_as_longlong(
        tex1Dfetch(J_xi, J_offset + 2 * xyz_id(x, y, z, L, L2)));
    spin_t Jn = int2_as_longlong(
        tex1Dfetch(J_yi, J_offset + 2 * xyz_id(x, y, z, L, L2)));
    spin_t Ju = int2_as_longlong(
        tex1Dfetch(J_zi, J_offset + 2 * xyz_id(x, y, z, L, L2)));
    spin_t Je = int2_as_longlong(
        tex1Dfetch(J_xi, J_offset + 2 * xyz_id(x, y, z, L, L2) + 1));
    spin_t Js = int2_as_longlong(
        tex1Dfetch(J_yi, J_offset + 2 * xyz_id(x, y, z, L, L2) + 1));
    spin_t Jd = int2_as_longlong(
        tex1Dfetch(J_zi, J_offset + 2 * xyz_id(x, y, z, L, L2) + 1));
    if (x != 0 && y != 0 && z != 0 && x != L - 1 && y != L - 1 && z != L - 1) {
      #ifdef SPLIT_H
      updat_spin_split(
          s_u[xyz_id(x, y, z, L, L2)], s_i[xyz_id(x - 1, y, z, L, L2)],
          s_i[xyz_id(x, y - 1, z, L, L2)], s_i[xyz_id(x, y, z - 1, L, L2)],
          s_i[xyz_id(x + 1, y, z, L, L2)], s_i[xyz_id(x, y + 1, z, L, L2)],
          s_i[xyz_id(x, y, z + 1, L, L2)], rand1, rand2, boltz, Jw, Jn, Ju, Je,
          Js, Jd);
        #else
        updat_spin(
            s_u[xyz_id(x, y, z, L, L2)], s_i[xyz_id(x - 1, y, z, L, L2)],
            s_i[xyz_id(x, y - 1, z, L, L2)], s_i[xyz_id(x, y, z - 1, L, L2)],
            s_i[xyz_id(x + 1, y, z, L, L2)], s_i[xyz_id(x, y + 1, z, L, L2)],
            s_i[xyz_id(x, y, z + 1, L, L2)], rand1, boltz, Jw, Jn, Ju, Je,
            Js, Jd);
        #endif
    } else {
      spin_t sw = (x != 0) ? s_i[xyz_id(x - 1, y, z, L, L2)]
                           : s_i[xyz_id(L - 1, y, z, L, L2)];
      spin_t sn = (y != 0) ? s_i[xyz_id(x, (y - 1), z, L, L2)]
                           : s_i[xyz_id(x, (L - 1), z, L, L2)];
      spin_t su = (z != 0) ? s_i[xyz_id(x, y, (z - 1), L, L2)]
                           : s_i[xyz_id(x, y, (L - 1), L, L2)];
      spin_t se = s_i[xyz_id((x + 1) % L, y, z, L, L2)];
      spin_t ss = s_i[xyz_id(x, (y + 1) % L, z, L, L2)];
      spin_t sd = s_i[xyz_id(x, y, (z + 1) % L, L, L2)];
      #ifdef SPLIT_H
      updat_spin_split(s_u[xyz_id(x, y, z, L, L2)], sw, sn, su, se, ss, sd,
      rand2,rand2, boltz, Jw, Jn, Ju, Je, Js, Jd);
      #else
      updat_spin(s_u[xyz_id(x, y, z, L, L2)], sw, sn, su, se, ss, sd,
      rand1, boltz, Jw, Jn, Ju, Je, Js, Jd);
      #endif
    }
  }
}

__global__ void J_order_3d(int2 *J_xi_d, int2 *J_yi_d, int2 *J_zi_d,
                           unsigned int *buffer, int L, long N) {
  long x = blockIdx.x * blockDim.x + threadIdx.x;
  long L2 = L * L;
  // printf("%d\n",buffer[2*3*N-1]);
  if (x < N) {
    J_xi_d[2 * x] = make_int2(buffer[2 * (3 * x)], buffer[2 * (3 * x) + 1]);
    J_xi_d[2 * x + 1] = (((x + 1) % L) == 0)
                            ? make_int2(buffer[2 * (3 * (x - L + 1))],
                                        buffer[2 * (3 * (x - L + 1)) + 1])
                            : make_int2(buffer[2 * (3 * (x + 1))],
                                        buffer[2 * (3 * (x + 1)) + 1]);
    J_yi_d[2 * x] =
        make_int2(buffer[2 * (3 * x + 1)], buffer[2 * (3 * x + 1) + 1]);
    J_yi_d[2 * x + 1] = (((x + L) % L2) < L)
                            ? make_int2(buffer[2 * (3 * (x - L2 + L) + 1)],
                                        buffer[2 * (3 * (x - L2 + L) + 1) + 1])
                            : make_int2(buffer[2 * (3 * (x + L) + 1)],
                                        buffer[2 * (3 * (x + L) + 1) + 1]);
    J_zi_d[2 * x] =
        make_int2(buffer[2 * (3 * x + 2)], buffer[2 * (3 * x + 2) + 1]);
    J_zi_d[2 * x + 1] = ((x + L2) >= N)
                            ? make_int2(buffer[2 * (3 * (x - N + L2) + 2)],
                                        buffer[2 * (3 * (x - N + L2) + 2) + 1])
                            : make_int2(buffer[2 * (3 * (x + L2) + 2)],
                                        buffer[2 * (3 * (x + L2) + 2) + 1]);
  }
}

__global__ void measure_EJ_M_3d(spin_t *s1, float *EJ_buf, float *M_buf, long N,
                                int L) {

  __shared__ spin_t si[2 * 256];
  __shared__ spin_t Ew[2 * 256];
  __shared__ spin_t En[2 * 256];
  __shared__ spin_t Eu[2 * 256];

  long id1 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  long id2 = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1;

  int w_id = 2 * (threadIdx.x % 32);

  int w_s = 32 * 2 * (threadIdx.x / 32);

  si[w_s + w_id] = 0;
  si[w_s + w_id + 1] = 0;
  Ew[w_s + w_id] = 0;
  Ew[w_s + w_id + 1] = 0;
  En[w_s + w_id] = 0;
  En[w_s + w_id + 1] = 0;
  Eu[w_s + w_id] = 0;
  Eu[w_s + w_id + 1] = 0;

  if (id1 < N) {
    long L2 = L * L;

    spin_t si1 = s1[id1];
    spin_t sw1 = int2_as_longlong(tex1Dfetch(J_xi, 2 * id1)) ^ si1 ^
                 s1[((id1 % L) == 0) ? (id1 + L - 1) : (id1 - 1)];
    spin_t sn1 = int2_as_longlong(tex1Dfetch(J_yi, 2 * id1)) ^ si1 ^
                 s1[((id1 % (L2)) < L) ? (id1 + L2 - L) : (id1 - L)];
    spin_t su1 = int2_as_longlong(tex1Dfetch(J_zi, 2 * id1)) ^ si1 ^
                 s1[(id1 < L2) ? (id1 + N - L2) : (id1 - L2)];

    spin_t si2 = s1[id2];
    spin_t sw2 = int2_as_longlong(tex1Dfetch(J_xi, 2 * id2)) ^ si2 ^
                 s1[((id2 % L) == 0) ? (id2 + L - 1) : (id2 - 1)];
    spin_t sn2 = int2_as_longlong(tex1Dfetch(J_yi, 2 * id2)) ^ si2 ^
                 s1[((id2 % (L2)) < L) ? (id2 + L2 - L) : (id2 - L)];
    spin_t su2 = int2_as_longlong(tex1Dfetch(J_zi, 2 * id2)) ^ si2 ^
                 s1[(id2 < L2) ? (id2 + N - L2) : (id2 - L2)];

    int j1;

    // printf("%x %2x\n",sn2);

#pragma unroll
    for (int i = 0; i < 64; i++) {
      j1 = (w_id + i) % 64;
      // j2=(w_id+i+1)%64;

      si[w_s + j1] |= ((si1 >> j1) & ((spin_t)1UL)) << w_id;
      Ew[w_s + j1] |= ((sw1 >> j1) & ((spin_t)1UL)) << w_id;
      En[w_s + j1] |= ((sn1 >> j1) & ((spin_t)1UL)) << w_id;
      Eu[w_s + j1] |= ((su1 >> j1) & ((spin_t)1UL)) << w_id;

      si[w_s + j1] |= ((si2 >> j1) & ((spin_t)1UL)) << w_id + 1;
      Ew[w_s + j1] |= ((sw2 >> j1) & ((spin_t)1UL)) << w_id + 1;
      En[w_s + j1] |= ((sn2 >> j1) & ((spin_t)1UL)) << w_id + 1;
      Eu[w_s + j1] |= ((su2 >> j1) & ((spin_t)1UL)) << w_id + 1;
    }
  }


  if (threadIdx.x < 64) {
    __syncthreads();
    int M = 0, E = 0;
    double over = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      M += __popcll(si[threadIdx.x + i * 64]);
      E += __popcll(Ew[threadIdx.x + i * 64]) +
           __popcll(En[threadIdx.x + i * 64]) +
           __popcll(Eu[threadIdx.x + i * 64]);
    }

    if (((blockIdx.x + 1) * 512) > N) {
      over = (blockIdx.x + 1) * 512 - N;
    }

    M_buf[threadIdx.x * gridDim.x + blockIdx.x] = (2. * M - 8 * 64) + over;
    EJ_buf[threadIdx.x * gridDim.x + blockIdx.x] =
        (2. * E - 24. * 64.) + 3 * over;

  }
}

__forceinline__ __device__ void updat_spin_split(
    spin_t &s_i, spin_t &s_w, spin_t &s_n, spin_t &s_u, spin_t &s_e,
    spin_t &s_s, spin_t &s_d, float &rand1, float &rand2, float *boltz,
    spin_t &Jw, spin_t &Jn, spin_t &Ju, spin_t &Je, spin_t &Js, spin_t &Jd) {
  spin_t sw, sn, se, ss, su, sd, p0, p1, p2, mask, d0, d1, d2, dh, mh;

  sw = Jw ^ s_i ^ s_w;
  sn = Jn ^ s_i ^ s_n;
  su = Ju ^ s_i ^ s_u;
  se = Je ^ s_i ^ s_e;
  ss = Js ^ s_i ^ s_s;
  sd = Jd ^ s_i ^ s_d;

  p0 = sw & se & sn & ss & su & sd;
  p1 = ((sw ^ se) & sn & ss & su & sd) | (sw & se & (sn ^ ss) & su & sd) |
       (sw & se & sn & ss & (su ^ sd));
  p2 = ((sw ^ se) & (sn ^ ss) & su & sd) | ((sw ^ se) & sn & ss & (su ^ sd)) |
       (sw & se & (sn ^ ss) & (su ^ sd)) | ((sw ^ sd) & (se ^ su) & sn & ss) |
       (sw & (se ^ sn) & (ss ^ su) & sd);

  mask = ~(spin_t)0;
  d0 = rand1 < boltz[2] ? mask : 0;
  d1 = rand1 < boltz[1] ? mask : 0;
  d2 = rand1 < boltz[0] ? mask : 0;
  dh = rand2 < boltz[3] ? mask : 0;

  mh = ~s_i | dh;

  s_i ^= ((p0 & d0) | (p1 & d1) | (p2 & d2) | ~(p0 | p1 | p2)) & mh;
}
__forceinline__ __device__ void updat_spin(spin_t &s_i,spin_t &s_w,spin_t &s_n,spin_t &s_u,spin_t &s_e,spin_t &s_s,spin_t &s_d,float &rand1,float *boltz,spin_t &Jw,spin_t &Jn,spin_t &Ju,spin_t &Je,spin_t &Js,spin_t &Jd){
  spin_t sw = Jw ^ s_i ^ s_w;
  spin_t sn = Jn ^ s_i ^ s_n;
  spin_t su = Ju ^ s_i ^ s_u;
  spin_t se = Je ^ s_i ^ s_e;
  spin_t ss = Js ^ s_i ^ s_s;
  spin_t sd = Jd ^ s_i ^ s_d;

  spin_t p0 = ~sw & ~se & ~sn & ~ss & ~su & ~sd;
  spin_t p1 = ((sw ^ se) & ~sn & ~ss & ~su & ~sd) |(~sw & ~se & (sn ^ ss) & ~su & ~sd) |(~sw & ~se & ~sn & ~ss & (su ^ sd));
  spin_t p2 = ((sw ^ se) & (sn ^ ss) & ~su & ~sd) |((sw ^ se) & ~sn & ~ss & (su ^ sd)) |(~sw & ~se & (sn ^ ss) & (su ^ sd)) |((sw ^ sd) & (se ^ su) & ~sn & ~ss) |(~sw & (se ^ sn) & (ss ^ su) & ~sd);
  spin_t p3 = ((sw ^ se) & (sn ^ ss) & (su ^ sd))|((sw ^ sd) & (sn ^ se) & (su ^ ss))|((sw ^ ss) & (sn ^ sd) & (su ^ se));
  spin_t p4 = ((sw ^ se) & (sn ^ ss) & su & sd) |((sw ^ se) & sn & ss & (su ^ sd)) |(sw & se & (sn ^ ss) & (su ^ sd)) |((sw ^ sd) & (se ^ su) & sn & ss) |(sw & (se ^ sn) & (ss ^ su) & sd);
  spin_t p5 = ((sw ^ se) & sn & ss & su & sd) |(sw & se & (sn ^ ss) & su & sd) |(sw & se & sn & ss & (su ^ sd));
  spin_t p6 = sw & se & sn & ss & su & sd;


  spin_t mask = ~(spin_t)0;

  spin_t d00 = rand1 < boltz[0] ? mask : 0;
  spin_t d10 = rand1 < boltz[1] ? mask : 0;
  spin_t d20 = rand1 < boltz[2] ? mask : 0;
  spin_t d30 = rand1 < boltz[3] ? mask : 0;
  spin_t d40 = rand1 < boltz[4] ? mask : 0;
  spin_t d50 = rand1 < boltz[5] ? mask : 0;
  spin_t d60 = rand1 < boltz[6] ? mask : 0;

  spin_t d01 = rand1 < boltz[0+7] ? mask : 0;
  spin_t d11 = rand1 < boltz[1+7] ? mask : 0;
  spin_t d21 = rand1 < boltz[2+7] ? mask : 0;
  spin_t d31 = rand1 < boltz[3+7] ? mask : 0;
  spin_t d41 = rand1 < boltz[4+7] ? mask : 0;
  spin_t d51 = rand1 < boltz[5+7] ? mask : 0;
  spin_t d61 = rand1 < boltz[6+7] ? mask : 0;

  spin_t h = s_i ;

  spin_t d0 =(d00 | h) & (d01 | ~h);
  spin_t d1 =(d10 | h) & (d11 | ~h);
  spin_t d2 =(d20 | h) & (d21 | ~h);
  spin_t d3 =(d30 | h) & (d31 | ~h);
  spin_t d4 =(d40 | h) & (d41 | ~h);
  spin_t d5 =(d50 | h) & (d51 | ~h);
  spin_t d6 =(d60 | h) & (d61 | ~h);

  s_i ^= (p0 & d0) | (p1 & d1) | (p2 & d2) | (p3 & d3) | (p4 & d4) | (p5 & d5) | (p6 & d6);
}


__global__ void checkerbord_switch_3d(spin_t *s_1, spin_t *s_2, int L, int L2) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int z = 2 * threadIdx.z + blockIdx.z * blockDim.z + x % 2;
  if (x < L && y < L && z < L) {
    spin_t b = s_1[xyz_id(x, y, z, L, L2)];
    s_1[xyz_id(x, y, z, L, L2)] = s_2[xyz_id(x, y, z, L, L2)];
    s_2[xyz_id(x, y, z, L, L2)] = b;
  }
}

__global__ void swap_3d(spin_t *s_a, spin_t *s_b, spin_t mask, long N) {
  long x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    swap_bits(s_a[x], s_b[x], mask);
  }
}
