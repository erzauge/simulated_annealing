#pragma once
#include <vector>
#include "Logging.hpp"

template <typename T>
std::vector<int>  max_count(std::vector<T> *x,unsigned int n){
  if (n<2) {
    LOG(LOG_ERROR)<<"n to smale";
  }
  std::vector<int> count(x[0].size(),1);
  std::vector<T> max=x[0];
  for (size_t i = 1; i < n; i++) {
    for (size_t j = 0; j < count.size(); j++) {
      if (max[j]<x[i][j]) {
        max[j]=x[i][j];
        count[j]=1;
      }
      else if(max[j]==x[i][j]){
        count[j]++;
      }
    }
  }
  return count;
}

template <typename T>
bool all_grater(std::vector<T> &x,T a){
  bool r=true;
  for (T const &i:x) {
    r=r&&(i>a);
  }
  return r;
}
