#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

using namespace std;

typedef unsigned long long spin_t;

class gpu_sys {
 public:
  virtual void sweep() = 0;
  virtual void set_T(double T_) = 0;
  virtual void set_h(double h_) = 0;
  virtual void set_seed(long long seed_) = 0;
  virtual vector<float> measure() = 0;
  virtual void gpu_free() = 0;
  virtual void load_J(string fname) = 0;
  virtual void save_J(string fname) = 0;
  virtual void init_J() = 0;
  virtual void init_rand() = 0;
  virtual void save_sys(string prefix) = 0;
  virtual long get_N() = 0;
  virtual void swap(gpu_sys* sys, std::unique_ptr<spin_t[]> mask) = 0;
  virtual ostream& save(ostream& stream) = 0;
  virtual istream& load(istream& stream) = 0;
};
