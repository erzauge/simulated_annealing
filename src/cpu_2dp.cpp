#include "cpu_2dp.hpp"
#include "Logging.hpp"
#include "bin_io.hpp"
#include "sys_file.hpp"
#include "tools_inline.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
// #include <omp.h>

using namespace std;

/**
 * @brief cpu_2dp consruktor
 * @details inizalisirt das system
 *
 * @param L_ system größe
 * @param T_ temperatur
 * @param h_ magentfeld
 */
cpu_2dp::cpu_2dp(int L_, double T_, double h_)
    : uni_double(0., 1.), uni_int(0, (L_ * L_) - 1) {
  L = L_;
  T = T_;
  h = h_;
  N = L * L;
  s = new spin_t[N];
  Jx = new spin_t[2 * N];
  Jy = new spin_t[2 * N];
  #ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  #else
  for (int i = 0; i < 4; ++i) {
    int j = 1 + i;
    boltz[i] = exp(-2. * j * (1. / T));
  }
  boltz[4] = exp(-2 * h / T);
  #endif


  gen.seed(1234);
}

cpu_2dp::cpu_2dp(cpu_2dp const &x):uni_double(0., 1.), uni_int(0, (x.L * x.L) - 1){
  L = x.L;
  T = x.T;
  h = x.h;
  N = x.N;
  s = new spin_t[N];
  Jx = new spin_t[2 * N];
  Jy = new spin_t[2 * N];
  #ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  #else
  for (int i = 0; i < 4; ++i) {
    int j = 1 + i;
    boltz[i] = exp(-2. * j * (1. / T));
  }
  boltz[4] = exp(-2 * h / T);
  #endif

  for (long i = 0; i < N; i++) {
    s[i]=x.s[i];
    Jx[2*i]=x.Jx[2*i];
    Jx[2*i+1]=x.Jx[2*i+1];
    Jy[2*i]=x.Jy[2*i];
    Jy[2*i+1]=x.Jy[2*i+1];
  }

  gen.seed(1234);
}

cpu_2dp::~cpu_2dp() {
  delete[] s;
  delete[] Jx;
  delete[] Jy;
}

/**
 * @brief set seed
 * @details set the seed of the random nuber gnerator
 *
 * @param se seed
 */
void cpu_2dp::set_seed(long se) { gen.seed(se); }

/**
 * @brief sweep system
 * @details simulate a sweep of the system system
 */
void cpu_2dp::sweep() {
  for (int offset = 0; offset < 2; offset++) {
    for (int y = 0; y < L; y++) {
      for (int x = (y + offset) % 2; x < L; x += 2) {
        update_spin(xy_id(x, y, L));
      }
    }
  }
}

/**
 * @brief set temperatur
 * @details set the tmperatur of the system
 *
 * @param T_ temperatur
 */
void cpu_2dp::set_T(double T_) {
  T = T_;
  #ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  // for (size_t i = 0; i < 20; i++) {
  //   LOG(LOG_ERROR)<<boltz[i]<<'\t'<<i;
  // }
  #else
  for (int i = 0; i < 4; ++i) {
    int j = 1 + i;
    boltz[i] = exp(-2. * j * (1. / T));
  }
  boltz[4] = exp(-2 * h / T);
  #endif
}

/**
 * @brief set mgnet feld
 * @details set the magnetc feld of the system
 *
 * @param h_ magnetic feld
 */
void cpu_2dp::set_h(double h_) {
  h = h_;
  #ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  #else
  for (int i = 0; i < 4; ++i) {
  boltz[4] = exp(-2 * h / T);
  #endif
}

/**
 * @brief measure magnetisaen
 * @details measure the magnetisaen of a system
 *
 * @param b select which system get measurd
 * @return magnetisaen
 */
void cpu_2dp::measure_M(double *r) {
  double M[64];
  for (int i = 0; i < 64; ++i) {
    M[i] = 0;
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < 64; ++j) {
      M[63 - j] -= bit_to_double(s[i], j);
    }
  }

  for (int i = 0; i < 64; ++i) {
    r[i] = M[i];
  }
}

/**
 * @brief measure  interection energy
 * @details measure the energy of a system
 *
 * @param b select which system get measurd
 * @return interection energy
 */
void cpu_2dp::measure_EJ(double *r) {
  double E[64];
  for (int i = 0; i < 64; ++i) {
    E[i] = 0;
  }

  for (int i = 0; i < N; ++i) {
    spin_t s_i = s[i];
    spin_t s_w = (i % L == 0) ? ~(spin_t)0 : Jx[2 * i] ^ s[(i - 1)] ^ s_i;
    spin_t s_n = Jy[2 * i] ^ s[(i < L) ? i + N - L : i - L] ^ s_i;
    for (int j = 0; j < 64; ++j) {
      E[63 - j] -= bit_to_double(s_w, j);
      E[63 - j] -= bit_to_double(s_n, j);
    }
  }
  for (int i = 0; i < 64; ++i) {
    r[i] = E[i] - L;
  }
}

// __builtin_popcountll()

/**
 * @brief initialize randomly
 * @details initialize system randome
 */
void cpu_2dp::init_rand() {
  for (int i = 0; i < N; ++i) {
    s[i] = gen();
  }
}

/**
 * @brief initialize orderd
 * @details initialize system orderd
 */
void cpu_2dp::init_order() {

  for (int i = 0; i < N; ++i) {
    s[i] = ~0;
  }
}

/**
 * @brief initialize J
 * @details initialize J randomly
 */
void cpu_2dp::init_J() {

  spin_t *buffer = new spin_t[2 * N];

  for (long i = 0; i < 2 * N; ++i) {
    buffer[i] = gen();
  }
  for (long x = 0; x < N; ++x) {
    Jx[2 * x] = buffer[2 * x];
    Jx[2 * x + 1] =
        (x % L == (L - 1)) ? buffer[2 * (x - L + 1)] : buffer[2 * (x + 1)];
    Jy[2 * x] = buffer[2 * x + 1];
    Jy[2 * x + 1] =
        ((x + L) >= N) ? buffer[2 * (x - N + L) + 1] : buffer[2 * (x + L) + 1];
  }

  delete[] buffer;
}

/**
 * @brief load J
 * @details loads interection constans
 *
 * @param fname filename
 */
void cpu_2dp::load_J(string fname) {
  int error = load_J_2d(Jx, Jy, N, fname);
  if (error < 0) {
    LOG(LOG_ERROR) << "fehler beim laden von J in file \"" << fname
                   << "\" mit fehler " << error;
  }
}

/**
 * @brief saves J
 * @details saves interection constans
 *
 * @param fname filename
 */
void cpu_2dp::save_J(string fname) {
  int error = save_J_2d(Jx, Jy, N, fname);
  if (error < 0) {
    LOG(LOG_ERROR) << "fehler beim laden von J in file \"" << fname
                   << "\" mit fehler " << error;
  }
}

/**
 * @brief measure energy
 * @details prints the energy of all systems
 */
vector<float> cpu_2dp::measure() {

  double E[64];
  double M[64];
  measure_M(M);
  measure_EJ(E);
  vector<float> result;
  result.assign(64, 0);
  for (int i = 0; i < 64; ++i) {
    result[i] = E[i] + h * M[i];
  }
  // LOG(LOG_DEBUG) << result.size();
  return result;
}

/**
 * @brief saves system
 * @details saves image of systems
 *
 * @param prefix file prefix
 */
void cpu_2dp::save_sys(string prefix) {
  for (int i = 0; i < 64; ++i) {
    stringstream str_i;
    str_i << i;
    // image setup
    ofstream file(prefix + str_i.str() + ".pbm");
    file << "P1" << endl;
    file << L << " " << L << endl;
    // print image
    for (int j = 0; j < L; ++j) {
      for (int k = 0; k < L; ++k) {
        file << (((s[j * L + k] & ((spin_t)1 << i)) == 0) ? "0 " : "1 ");
      }
      file << endl;
    }
  }
}

long cpu_2dp::get_N() { return N; }

#ifndef SPLIT_H
inline void cpu_2dp::update_spin(long s_int) {
  spin_t s_i = s[s_int];
  spin_t s_w = ((s_int % L == 0) ? ~(spin_t)0 : Jx[2 * s_int] ^ s[(s_int - 1)] ^ s_i);
  spin_t s_e = (((s_int + 1) % L == 0) ? ~(spin_t)0 : Jx[2 * s_int + 1] ^ s[s_int + 1] ^ s_i);
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
  if ((s_int % L == 0) || (((s_int + 1) % L) == 0)) {// nicht perodischer rand
    d00 = rand1 < boltz[0+10]? mask : (spin_t)0;
    d10 = rand1 < boltz[1+10]? mask : (spin_t)0;
    d20 = rand1 < boltz[2+10]? mask : (spin_t)0;
    d30 = rand1 < boltz[3+10]? mask : (spin_t)0;
    d40 = rand1 < boltz[4+10]? mask : (spin_t)0;
    d01 = rand1 < boltz[0+5+10]? mask : (spin_t)0;
    d11 = rand1 < boltz[1+5+10]? mask : (spin_t)0;
    d21 = rand1 < boltz[2+5+10]? mask : (spin_t)0;
    d31 = rand1 < boltz[3+5+10]? mask : (spin_t)0;
    d41 = rand1 < boltz[4+5+10]? mask : (spin_t)0;
  } else {
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
  }
  spin_t h =s_i;
  spin_t d0 = (d00 | h) & (d01 | ~h);
  spin_t d1 = (d10 | h) & (d11 | ~h);
  spin_t d2 = (d20 | h) & (d21 | ~h);
  spin_t d3 = (d30 | h) & (d31 | ~h);
  spin_t d4 = (d40 | h) & (d41 | ~h);

  s[s_int] ^= (d0&p0)|(d1&p1)|(d2&p2)|(d3&p3)|(d4&p4);
}
#else
inline void cpu_2dp::update_spin(long s_int) {
  spin_t s_i = s[s_int];
  spin_t s_w =
  (s_int % L == 0) ? ~(spin_t)0 : Jx[2 * s_int] ^ s[(s_int - 1)] ^ s_i;
  spin_t s_e = ((s_int + 1) % L == 0) ? ~(spin_t)0
  : Jx[2 * s_int + 1] ^ s[s_int + 1] ^ s_i;
  spin_t s_n = Jy[2 * s_int] ^ s[(s_int < L) ? s_int + N - L : s_int - L] ^ s_i;
  spin_t s_s = Jy[2 * s_int + 1] ^
  s[((s_int + L) < N) ? s_int + L : s_int + L - N] ^ s_i;

  spin_t p0 = s_w & s_n & s_e & s_s;
  spin_t p1 = ((s_w ^ s_n) & s_e & s_s) | (s_w & s_n & (s_e ^ s_s));
  spin_t border;
  if ((s_int % L == 0) || (((s_int + 1) % L) == 0)) {
    border = ~(spin_t)0;
  } else {
    border = (spin_t)0;
  }

  spin_t mask = ~(spin_t)0;
  double rand1 = uni_double(gen);
  double rand2 = uni_double(gen);

  spin_t d01 = rand1 < boltz[3 - 1] ? mask : (spin_t)0;  // nicht perodischer
  // rand
  spin_t d11 = rand1 < boltz[1 - 1] ? mask : (spin_t)0;  // nicht perodischer
  // rand
  spin_t d00 = rand1 < boltz[4 - 1] ? mask : (spin_t)0;
  spin_t d10 = rand1 < boltz[2 - 1] ? mask : (spin_t)0;
  spin_t dh = rand2 < boltz[4] ? mask : (spin_t)0;

  spin_t d0 = (d00 | border) & (d01 | ~border);
  spin_t d1 = (d10 | border) & (d11 | ~border);

  spin_t mh = ~s[s_int] | dh;

  s[s_int] ^= (((p0 & d0) | (p1 & d1) | ~(p0 | p1)) & mh);
}
#endif


void cpu_2dp::swap(cpu_sys *sys, std::unique_ptr<spin_t[]> mask) {
  cpu_2dp *sys_2d = dynamic_cast<cpu_2dp *>(sys);
  if (sys_2d != NULL) {
    swap(sys_2d, mask[0]);
  } else {
    LOG(LOG_ERROR) << "conversion error";
  }
}

void cpu_2dp::swap(cpu_2dp *sys, spin_t mask) {
  // LOG(LOG_INFO)<<sys->N;
  if (N == sys->N) {
    for (int i = 0; i < N; ++i) {
      swap_bits(s[i], sys->s[i], mask);
    }
  }
}

ostream &cpu_2dp::save(ostream &stream) {
  binary_write(stream, L, 1);
  binary_write(stream, T, 1);
  binary_write(stream, h, 1);
  binary_write(stream, s[0], N);
  binary_write(stream, Jx[0], 2 * N);
  binary_write(stream, Jy[0], 2 * N);
  binary_write(stream, gen, 1);
  binary_write(stream, uni_double, 1);
  binary_write(stream, uni_int, 1);
  return stream;
}

istream &cpu_2dp::load(istream &stream) {
  binary_read(stream, L, 1);
  N = L * L;
  binary_read(stream, T, 1);
  set_T(T);
  binary_read(stream, h, 1);
  set_h(h);
  delete[] s;
  delete[] Jx;
  delete[] Jy;
  s = new spin_t[N];
  Jx = new spin_t[2 * N];
  Jy = new spin_t[2 * N];
  binary_read(stream, s[0], N);
  binary_read(stream, Jx[0], 2 * N);
  binary_read(stream, Jy[0], 2 * N);
  binary_read(stream, gen, 1);  // dose thate worke?
  binary_read(stream, uni_double, 1);
  binary_read(stream, uni_int, 1);
  return stream;
}

double cpu_2dp::get_T() { return T; }

cpu_2dp * cpu_2dp::clone() const{
  return new cpu_2dp(*this);
}
