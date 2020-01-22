#pragma once
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <memory>

#include "cpu_sys.hpp"

using namespace std;

typedef unsigned long long spin_t;

inline int xy_id(int x, int y, int L) { return y * L + x; }

/**
 * @brief 2d system
 * @details cpu impelemtation of the 2d plan system
 */
class cpu_2dp : public cpu_sys {

protected:
  spin_t *s;
  spin_t *Jx;
  spin_t *Jy;
  double T;
  double h;
  #ifndef SPLIT_H
  double boltz[20];
  #else
  double boltz[5];
  #endif
  int L;
  long N;
  mt19937_64 gen;
  uniform_real_distribution<double> uni_double;
  uniform_int_distribution<int> uni_int;
  virtual void update_spin(long s_int);

public:
  cpu_2dp(int L_, double T_, double h_);
  cpu_2dp(cpu_2dp const & x);
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
  void set_h(double h_) ;
  void measure_M(double *r);
  void measure_EJ(double *r);
  vector<float> measure() override;
  void save_sys(string prefix) override;
  long get_N() override;
  void swap(cpu_sys *sys, std::unique_ptr<spin_t[]> mask) override;
  void swap(cpu_2dp *sys, spin_t mask);
  double get_T() override;
  cpu_2dp * clone() const;
  virtual ~cpu_2dp();
};
