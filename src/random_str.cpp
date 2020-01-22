#include "random_str.hpp"
#include <random>

std::string random_string(int n) {
  std::mt19937 gen{std::random_device{}()};
  std::uniform_int_distribution<short> dist{'a', 'z'};

  std::string str(n, '\0');
  for (auto& c : str) {
    c = dist(gen);
  }
  return str;
}
