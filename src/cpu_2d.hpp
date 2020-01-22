#pragma once
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <memory>

#include "cpu_2dp.hpp"

using namespace std;

class cpu_2d : public cpu_2dp {
protected:
  void update_spin(long s_int) override;
public:
  vector<float> measure() override;
  using cpu_2dp::cpu_2dp;
};
