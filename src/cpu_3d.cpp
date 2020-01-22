#include "cpu_3d.hpp"
#include "Logging.hpp"
#include "bin_io.hpp"
#include "sys_file.hpp"
#include "tools_inline.hpp"
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace std;

/**
 * @brief constructor
 * @details consturcto sets output stream to cout
 *
 * @param L_ system size
 * @param T_ temperatur
 * @param h_ magnetik feald
 */
cpu_3d::cpu_3d(int L_, double T_, double h_)
    : gen(1234), uni_double(0., 1.), uni_int(0, (L_ * L_ * L_) - 1) {
  L = L_;
  T = T_;
  h = h_;
  L2 = L * L;
  N = L * L * L;
  s = new spin_t[N];
  Jx = new spin_t[2 * N];
  Jy = new spin_t[2 * N];
  Jz = new spin_t[2 * N];
  #ifndef SPLIT_H
  for (size_t i = 0; i < 7; i++) {
    int j=(i-3)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
  }
  #else
  boltz[0] = exp((-2. * 6) / T);
  boltz[1] = exp((-2. * 4) / T);
  boltz[2] = exp((-2. * 2) / T);
  boltz[3] = exp((-2. * h) / T);
  #endif
  gen.seed(1234);
}

cpu_3d::cpu_3d(cpu_3d const & x): gen(1234), uni_double(0., 1.), uni_int(0, (x.N) - 1){
  L = x.L;
  T = x.T;
  h = x.h;
  L2 = x.L2;
  N = x.N;
  s = new spin_t[N];
  Jx = new spin_t[2 * N];
  Jy = new spin_t[2 * N];
  Jz = new spin_t[2 * N];
  #ifndef SPLIT_H
  for (size_t i = 0; i < 7; i++) {
    int j=(i-3)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
  }
  #else
  boltz[0] = exp((-2. * 6) / T);
  boltz[1] = exp((-2. * 4) / T);
  boltz[2] = exp((-2. * 2) / T);
  boltz[3] = exp((-2. * h) / T);
  #endif

  for (long i = 0; i < N; i++) {
    s[i]=x.s[i];
    Jx[2*i]=x.Jx[2*i];
    Jx[2*i+1]=x.Jx[2*i+1];
    Jy[2*i]=x.Jy[2*i];
    Jy[2*i+1]=x.Jy[2*i+1];
    Jz[2*i]=x.Jz[2*i];
    Jz[2*i+1]=x.Jz[2*i+1];
  }
  gen.seed(1234);
}

cpu_3d::~cpu_3d() {

  delete[] s;
  delete[] Jx;
  delete[] Jy;
  delete[] Jz;
}

/**
 * @brief sweep system
 * @details simulate a sweep of the system
 */
void cpu_3d::sweep() {

  for (int offset = 0; offset < 2; offset++) {
    for (int z = 0; z < L; z++) {
      for (int y = 0; y < L; y++) {
        for (int x = (y + offset) % 2; x < L; x += 2) {
          update_spin(xyz_id(x, y, z, L, L2));
        }
      }
    }
  }
}
#ifndef SPLIT_H
inline void cpu_3d::update_spin(long s_int) {
  spin_t s_i = s[s_int];
  spin_t s_w = Jx[2 * s_int] ^ s_i ^
  (((s_int % L) == 0) ? s[s_int + L - 1] : s[s_int - 1]);
  spin_t s_e = Jx[2 * s_int + 1] ^ s_i ^
  ((((s_int + 1) % L) == 0) ? s[s_int - L + 1] : s[s_int + 1]);
  spin_t s_n = Jy[2 * s_int] ^ s_i ^
  ((s_int % L2 < L) ? s[s_int + L2 - L] : s[s_int - L]);
  spin_t s_s = Jy[2 * s_int + 1] ^ s_i ^
  (((s_int + L) % L2 < L) ? s[s_int - L2 + L] : s[s_int + L]);
  spin_t s_u = Jz[2 * s_int] ^ s_i ^
  (((s_int - L2) < 0) ? s[s_int + N - L2] : s[s_int - L2]);
  spin_t s_d = Jz[2 * s_int + 1] ^ s_i ^
  (((s_int + L2) >= N) ? s[s_int - N + L2] : s[s_int + L2]);

  spin_t p0 = ~s_w & ~s_e & ~s_n & ~s_s & ~s_u & ~s_d;
  spin_t p1 = ((s_w ^ s_e) & ~s_n & ~s_s & ~s_u & ~s_d) |(~s_w & ~s_e & (s_n ^ s_s) & ~s_u & ~s_d) |(~s_w & ~s_e & ~s_n & ~s_s & (s_u ^ s_d));
  spin_t p2 = ((s_w ^ s_e) & (s_n ^ s_s) & ~s_u & ~s_d) |((s_w ^ s_e) & ~s_n & ~s_s & (s_u ^ s_d)) |(~s_w & ~s_e & (s_n ^ s_s) & (s_u ^ s_d)) |((s_w ^ s_d) & (s_e ^ s_u) & ~s_n & ~s_s) |(~s_w & (s_e ^ s_n) & (s_s ^ s_u) & ~s_d);
  spin_t p3 = ((s_w ^ s_e) & (s_n ^ s_s) & (s_u ^ s_d))|((s_w ^ s_d) & (s_n ^ s_e) & (s_u ^ s_s))|((s_w ^ s_s) & (s_n ^ s_d) & (s_u ^ s_e));
  spin_t p4 = ((s_w ^ s_e) & (s_n ^ s_s) & s_u & s_d) |((s_w ^ s_e) & s_n & s_s & (s_u ^ s_d)) |(s_w & s_e & (s_n ^ s_s) & (s_u ^ s_d)) |((s_w ^ s_d) & (s_e ^ s_u) & s_n & s_s) |(s_w & (s_e ^ s_n) & (s_s ^ s_u) & s_d);
  spin_t p5 = ((s_w ^ s_e) & s_n & s_s & s_u & s_d) |(s_w & s_e & (s_n ^ s_s) & s_u & s_d) |(s_w & s_e & s_n & s_s & (s_u ^ s_d));
  spin_t p6 = s_w & s_e & s_n & s_s & s_u & s_d;

  double rand1 = uni_double(gen);

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

  s[s_int] ^= (p0 & d0) | (p1 & d1) | (p2 & d2) | (p3 & d3) | (p4 & d4) | (p5 & d5) | (p6 & d6);
}
#else
inline void cpu_3d::update_spin(long s_int) {
  spin_t s_i = s[s_int];
  spin_t s_w = Jx[2 * s_int] ^ s_i ^
  (((s_int % L) == 0) ? s[s_int + L - 1] : s[s_int - 1]);
  spin_t s_e = Jx[2 * s_int + 1] ^ s_i ^
  ((((s_int + 1) % L) == 0) ? s[s_int - L + 1] : s[s_int + 1]);
  spin_t s_n = Jy[2 * s_int] ^ s_i ^
  ((s_int % L2 < L) ? s[s_int + L2 - L] : s[s_int - L]);
  spin_t s_s = Jy[2 * s_int + 1] ^ s_i ^
  (((s_int + L) % L2 < L) ? s[s_int - L2 + L] : s[s_int + L]);
  spin_t s_u = Jz[2 * s_int] ^ s_i ^
  (((s_int - L2) < 0) ? s[s_int + N - L2] : s[s_int - L2]);
  spin_t s_d = Jz[2 * s_int + 1] ^ s_i ^
  (((s_int + L2) >= N) ? s[s_int - N + L2] : s[s_int + L2]);

  spin_t p0 = s_w & s_e & s_n & s_s & s_u & s_d;
  spin_t p1 = ((s_w ^ s_e) & s_n & s_s & s_u & s_d) |
  (s_w & s_e & (s_n ^ s_s) & s_u & s_d) |
  (s_w & s_e & s_n & s_s & (s_u ^ s_d));
  spin_t p2 = ((s_w ^ s_e) & (s_n ^ s_s) & s_u & s_d) |
  ((s_w ^ s_e) & s_n & s_s & (s_u ^ s_d)) |
  (s_w & s_e & (s_n ^ s_s) & (s_u ^ s_d)) |
  ((s_w ^ s_d) & (s_e ^ s_u) & s_n & s_s) |
  (s_w & (s_e ^ s_n) & (s_s ^ s_u) & s_d);

  double rand1 = uni_double(gen);
  double rand2 = uni_double(gen);

  spin_t mask = ~(spin_t)0;

  spin_t d0 = rand1 < boltz[0] ? mask : 0;
  spin_t d1 = rand1 < boltz[1] ? mask : 0;
  spin_t d2 = rand1 < boltz[2] ? mask : 0;
  spin_t dh = rand2 < boltz[3] ? mask : 0;

  spin_t mh = ~s[s_int] | dh;

  s[s_int] ^= ((p0 & d0) | (p1 & d1) | (p2 & d2) | ~(p0 | p1 | p2)) & mh;
}
#endif

/**
 * @brief set seed
 * @details set the seed
 *
 * @param se seed
 */
void cpu_3d::set_seed(long se) { gen.seed(se); }

/**
 * @brief set temperatur
 * @details set the temperatur of the system
 *
 * @param T_ temperatur
 */
void cpu_3d::set_T(double T_) {
  T = T_;
  #ifndef SPLIT_H
  for (size_t i = 0; i < 7; i++) {
    int j=(i-3)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
  }
  #else
  boltz[0] = exp((-2. * 6) / T);
  boltz[1] = exp((-2. * 4) / T);
  boltz[2] = exp((-2. * 2) / T);
  boltz[3] = exp((-2. * h) / T);
  #endif
}

/**
 * @brief set magnetic feld
 * @details set magnetic feld of the system
 *
 * @param h_ magnetic feld
 */
void cpu_3d::set_h(double h_) {
  h = h_;
  #ifndef SPLIT_H
  for (size_t i = 0; i < 7; i++) {
    int j=(i-3)*2;
    boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
  }
  #else
  boltz[3] = exp((-2. * h) / T);
  #endif
}

/**
 * @brief initialize randomly
 * @details initialize system randome
 */
void cpu_3d::init_rand() {
  uniform_int_distribution<int> dist(0, 1);

  for (int i = 0; i < N; ++i) {
    s[i] = (spin_t)gen();
  }
}

/**
 * @brief initialize orderd
 * @details initialize system orderd
 */
void cpu_3d::init_order() {
  uniform_int_distribution<int> dist(0, 1);

  for (int i = 0; i < N; ++i) {
    s[i] = ~0;
  }
}

/**
 * @brief initialize J
 * @details initialize J randomly
 */
void cpu_3d::init_J() {

  spin_t *buffer = new spin_t[3 * N];

  for (int i = 0; i < 3 * N; ++i) {
    buffer[i] = (spin_t)gen();
  }

  for (int x = 0; x < N; ++x) {
    Jx[2 * x] = buffer[3 * x];
    Jx[2 * x + 1] =
        (((x + 1) % L == 0) ? buffer[3 * (x - L + 1)] : buffer[3 * (x + 1)]);
    Jy[2 * x] = buffer[3 * x + 1];
    Jy[2 * x + 1] = (((x + L) % L2 < L) ? buffer[3 * (x - L2 + L) + 1]
                                        : buffer[3 * (x + L) + 1]);
    Jz[2 * x] = buffer[3 * x + 2];
    Jz[2 * x + 1] = (((x + L2) >= N) ? buffer[3 * (x - N + L2) + 2]
                                     : buffer[3 * (x + L2) + 2]);
  }
  delete[] buffer;
}

/**
 * @brief measure magnetisaen
 * @details measure the magnetisaen of a system
 *
 * @param b select which system get measurd
 * @return magnetisaen
 */
void cpu_3d::measure_M(double *r) {
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
void cpu_3d::measure_EJ(double *r) {

  double E[64];
  for (int i = 0; i < 64; ++i) {
    E[i] = 0;
  }

  for (int i = 0; i < N; ++i) {
    spin_t s_i = s[i];
    spin_t s_w = Jx[2 * i] ^ s_i ^ ((i % L) == 0 ? s[i + L - 1] : s[i - 1]);
    spin_t s_n = Jy[2 * i] ^ s_i ^ ((i % L2 < L) ? s[i + L2 - L] : s[i - L]);
    spin_t s_u = Jz[2 * i] ^ s_i ^ ((i - L2) < 0 ? s[i + N - L2] : s[i - L2]);
    for (int j = 0; j < 64; ++j) {
      E[63 - j] -= bit_to_double(s_w, j);
      E[63 - j] -= bit_to_double(s_n, j);
      E[63 - j] -= bit_to_double(s_u, j);
    }
  }
  for (int i = 0; i < 64; ++i) {
    r[i] = E[i];
  }
}

/**
 * @brief measure energy
 * @details prints the energy of all systems
 */
vector<float> cpu_3d::measure() {
  double E[64];
  double M[64];
  measure_M(M);
  measure_EJ(E);
  vector<float> result;
  result.assign(64, 0);
  for (int i = 0; i < 64; ++i) {
    result[i] = E[i] + h * M[i];
  }
  return result;
}

/**
 * @brief load J
 * @details loads interection constans
 *
 * @param fname filename
 */
void cpu_3d::load_J(string fname) {
  int error = load_J_3d(Jx, Jy, Jz, N, fname);
  if (error < 0) {
    LOG(LOG_ERROR) << "fehler beim laden von J in file \"" << fname
                   << "\" mit fehler " << error << endl;
  }
}

/**
 * @brief saves J
 * @details saves interection constans
 *
 * @param fname filename
 */
void cpu_3d::save_J(string fname) {
  int error = save_J_3d(Jx, Jy, Jz, N, fname);
  if (error < 0) {
    LOG(LOG_ERROR) << "fehler beim laden von J in file \"" << fname
                   << "\" mit fehler " << error << endl;
  }
}

/**
 * @brief saves system
 * @details saves image of systems
 *
 * @param prefix file prefix
 */
void cpu_3d::save_sys(string prefix) {
  for (int i = 0; i < 64; ++i) {
    stringstream str_i;
    str_i << i;
    // image setup
    ofstream file(prefix + str_i.str() + ".pbm");
    file << "P1" << endl;
    file << L << " " << L * L << endl;
    // print image
    for (int j = 0; j < L * L; ++j) {
      for (int k = 0; k < L; ++k) {
        file << (((s[j * L + k] & ((spin_t)1 << i)) == 0)
                     ? "0 "
                     : "1 ");  // muss ich noch mal überprüfen
      }
      file << endl;
    }
  }
}

long cpu_3d::get_N() { return N; }

void cpu_3d::swap(cpu_sys *sys, std::unique_ptr<spin_t[]> mask) {
  cpu_3d *sys_3d = dynamic_cast<cpu_3d *>(sys);
  if (sys_3d != NULL) {
    swap(sys_3d, mask[0]);
  } else {
    LOG(LOG_ERROR) << "conversion error";
  }
}

void cpu_3d::swap(cpu_3d *sys, spin_t mask) {
  if (N == sys->N) {
    for (int i = 0; i < N; ++i) {
      swap_bits(s[i], sys->s[i], mask);
    }
  }
}
/*!
   \brief saves stade of objeckt in stream
   \param stream file stream to save the stade in.
*/
ostream & cpu_3d::save(ostream &stream) {
  binary_write(stream, L, 1);
  binary_write(stream, T, 1);
  binary_write(stream, h, 1);
  binary_write(stream, s[0], N);
  binary_write(stream, Jx[0], 2 * N);
  binary_write(stream, Jy[0], 2 * N);
  binary_write(stream, Jz[0], 2 * N);
  binary_write(stream, gen, 1);
  binary_write(stream, uni_double, 1);
  binary_write(stream, uni_int, 1);
  return stream;
}

/*!
   \brief loads the stade of objeckt from stream
   \param stream file stream where the stade gets loaded from.
*/
istream & cpu_3d::load(istream &stream) {
  binary_read(stream, L, 1);
  L2 = L * L;
  N = L2 * L;
  binary_read(stream, T, 1);
  set_T(T);
  binary_read(stream, h, 1);
  set_h(h);
  delete[] s;
  delete[] Jx;
  delete[] Jy;
  delete[] Jz;
  s = new spin_t[N];
  Jx = new spin_t[2 * N];
  Jy = new spin_t[2 * N];
  Jz = new spin_t[2 * N];
  binary_read(stream, s[0], N);
  binary_read(stream, Jx[0], 2 * N);
  binary_read(stream, Jy[0], 2 * N);
  binary_read(stream, Jz[0], 2 * N);
  binary_read(stream, gen, 1);  // dose thate worke?
  binary_read(stream, uni_double, 1);
  binary_read(stream, uni_int, 1);
  return stream;
}

double cpu_3d::get_T() { return T; }

cpu_3d * cpu_3d::clone() const{
  return new cpu_3d(*this);
}
