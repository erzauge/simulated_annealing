#pragma once
#include <random>
#include <string>
#include <vector>
#include <memory>

#include "cpu_sys.hpp"

using namespace std;

inline long xyz_id(int x, int y, int z, int L, int L2) {
  return z * L2 + y * L + x;
}

typedef unsigned long long spin_t;

/**
 * @brief 2d system
 * @details cpu impelemtation of the 3d system
 */
class cpu_3d : public cpu_sys {
 private:
  spin_t *s;
  spin_t *Jx;
  spin_t *Jy;
  spin_t *Jz;
  double T;
  double h;
  #ifndef SPLIT_H
  double boltz[14];
  #else
  double boltz[4];
  #endif
  int L;
  long L2;
  long N;
  mt19937_64 gen;
  // mt19937 gen;
  uniform_real_distribution<double> uni_double;
  uniform_int_distribution<int> uni_int;
  void update_spin(long s_int);

 public:
  cpu_3d(int L_, double T_, double h_);
  cpu_3d(cpu_3d const & x);
  ostream & save(ostream &stream) override;
  istream & load(istream &stream) override;
  void set_seed(long se) override;
  void init_rand() override;
  void init_order() override;
  void init_J() override;
  void load_J(string fname) override;
  void save_J(string fname) override;
  void sweep() override;
  void set_T(double T_) override;
  void set_h(double h_);
  void measure_M(double *r);
  void measure_EJ(double *r);
  vector<float> measure() override;
  void save_sys(string prefix) override;
  long get_N() override;
  void swap(cpu_sys *sys, std::unique_ptr<spin_t[]>) override;
  void swap(cpu_3d *sys, spin_t mask);
  double get_T() override;
  cpu_3d * clone() const;
  virtual ~cpu_3d();
};
