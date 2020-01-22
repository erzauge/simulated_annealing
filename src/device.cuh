#pragma once
typedef unsigned long long spin_t;

__forceinline__ __device__ unsigned long long int2_as_longlong(int2 a) {
  unsigned long long res;
  asm("mov.b64 %0, {%1,%2};" : "=l"(res) : "r"(a.x), "r"(a.y));
  return res;
}

__forceinline__ __device__ unsigned long long int2_as_longlong(unsigned int a,
                                                               unsigned int b) {
  unsigned long long res;
  asm("mov.b64 %0, {%1,%2};" : "=l"(res) : "r"(a), "r"(b));
  return res;
}

// inspierd by http://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
__forceinline__ __device__ void swap_bits(spin_t &a, spin_t &b, spin_t &mask) {
  spin_t r_a = a ^ ((a ^ b) & mask);
  spin_t r_b = b ^ ((b ^ a) & mask);
  a = r_a;
  b = r_b;
}
