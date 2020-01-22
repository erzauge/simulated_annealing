#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

using namespace std;

typedef unsigned long long spin_t;

class cpu_sys {
 public:
  virtual void set_seed(long se) = 0;
  virtual void init_rand() = 0;
  virtual void init_order() = 0;
  virtual void init_J() = 0;
  virtual void load_J(string fname) = 0;
  virtual void save_J(string fname) = 0;
  virtual void sweep() = 0;
  virtual void set_T(double T_) = 0;
  virtual vector<float> measure() = 0;
  virtual void save_sys(string prefix) = 0;
  virtual long get_N() = 0;
  virtual double get_T() = 0;
  virtual void swap(cpu_sys* sys, std::unique_ptr<spin_t[]>) = 0;
  virtual ostream& save(ostream& stream) = 0;
  virtual istream& load(istream& stream) = 0;
  virtual cpu_sys * clone() const = 0;
  virtual ~cpu_sys() = default;
};
