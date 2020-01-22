#include "cpu_2d.hpp"
#include "tools_inline.hpp"

#ifndef SPLIT_H
inline void cpu_2d::update_spin(long s_int) {
  spin_t s_i = s[s_int];
  spin_t s_w = (Jx[2 * s_int] ^ s[(s_int % L == 0) ? (s_int + L - 1) :(s_int - 1)] ^ s_i);
  spin_t s_e = (Jx[2 * s_int + 1] ^ s[((s_int + 1) % L == 0) ? s_int - L + 1 :s_int + 1] ^ s_i);
  spin_t s_n = (Jy[2 * s_int] ^ s[(s_int < L) ? s_int + N - L : s_int - L] ^ s_i);
  spin_t s_s = (Jy[2 * s_int + 1] ^ s[((s_int + L) < N) ? s_int + L : s_int + L - N] ^ s_i);

  spin_t p0 = ((~s_w)&(~s_e)&(~s_n)&~s_s);
  spin_t p1 = ((s_w^s_n)&~s_e&~s_s)|(~s_w&~s_n&(s_e^s_s));
  spin_t p2 = ((s_w ^ s_n)&(s_e ^ s_s))|((s_w ^ s_s)&(s_e ^ s_n));
  spin_t p3 = ((s_w^s_n)&s_e&s_s)|(s_w&s_n&(s_e^s_s));
  spin_t p4 = s_w&s_e&s_n&s_s;

  spin_t mask = ~(spin_t)0;
  double rand1 = uni_double(gen);
  spin_t d00 ,d10 ,d20 ,d30 ,d40 ,d01 ,d11 ,d21 ,d31 ,d41;

  d00 = rand1 < boltz[0]? mask : (spin_t)0;
  d10 = rand1 < boltz[1]? mask : (spin_t)0;
  d20 = rand1 < boltz[2]? mask : (spin_t)0;
  d30 = rand1 < boltz[3]? mask : (spin_t)0;
  d40 = rand1 < boltz[4]? mask : (spin_t)0;
  d01 = rand1 < boltz[0+5]? mask : (spin_t)0;
  d11 = rand1 < boltz[1+5]? mask : (spin_t)0;
  d21 = rand1 < boltz[2+5]? mask : (spin_t)0;
  d31 = rand1 < boltz[3+5]? mask : (spin_t)0;
  d41 = rand1 < boltz[4+5]? mask : (spin_t)0;

  spin_t h =s_i;
  spin_t d0 = (d00 | h) & (d01 | ~h);
  spin_t d1 = (d10 | h) & (d11 | ~h);
  spin_t d2 = (d20 | h) & (d21 | ~h);
  spin_t d3 = (d30 | h) & (d31 | ~h);
  spin_t d4 = (d40 | h) & (d41 | ~h);

  s[s_int] ^= (d0&p0)|(d1&p1)|(d2&p2)|(d3&p3)|(d4&p4);
}
#else
inline void cpu_2d::update_spin(long s_int) {
  spin_t s_i = s[s_int];
  spin_t s_w = Jx[2 * s_int] ^ s[(s_int % L == 0) ?  (s_int + L - 1) : (s_int - 1)] ^ s_i;
  spin_t s_e = Jx[2 * s_int + 1] ^ s[((s_int + 1) % L == 0) ? s_int - L + 1 : s_int + 1] ^ s_i;
  spin_t s_n = Jy[2 * s_int] ^ s[(s_int < L) ? s_int + N - L : s_int - L] ^ s_i;
  spin_t s_s = Jy[2 * s_int + 1] ^ s[((s_int + L) < N) ? s_int + L : s_int + L - N] ^ s_i;

  spin_t p0 = s_w & s_n & s_e & s_s;
  spin_t p1 = ((s_w ^ s_n) & s_e & s_s) | (s_w & s_n & (s_e ^ s_s));

  spin_t mask = ~(spin_t)0;
  double rand1 = uni_double(gen);
  double rand2 = uni_double(gen);

  spin_t d0 = rand1 < boltz[4 - 1] ? mask : (spin_t)0;
  spin_t d1 = rand1 < boltz[2 - 1] ? mask : (spin_t)0;
  spin_t dh = rand2 < boltz[4] ? mask : (spin_t)0;

  spin_t mh = ~s[s_int] | dh;

  s[s_int] ^= (((p0 & d0) | (p1 & d1) | ~(p0 | p1)) & mh);
}
#endif

vector<float> cpu_2d::measure() {

  double E[64];
  double M[64];

  for (int i = 0; i < 64; ++i) {
    E[i] = 0;
    M[i] = 0;
  }

  for (int i = 0; i < N; ++i) {
    spin_t s_i = s[i];
    spin_t s_w =  Jx[2 * i] ^ s[(i % L == 0) ? (i + L - 1) :(i - 1)] ^ s_i;
    spin_t s_n = Jy[2 * i] ^ s[(i < L) ? i + N - L : i - L] ^ s_i;
    for (int j = 0; j < 64; ++j) {
      E[63 - j] -= bit_to_double(s_w, j);
      E[63 - j] -= bit_to_double(s_n, j);
      M[63 - j] -= bit_to_double(s[i], j);
    }
  }

  vector<float> result;
  result.assign(64, 0);
  for (int i = 0; i < 64; ++i) {
    result[i] = E[i] + h * M[i];
  }
  return result;
}
