#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <vector>

#include "Logging.hpp"
#include "accumulator.hpp"
#include "bin_io.hpp"
#include "tools_inline.hpp"

typedef unsigned long long spin_t;

template <class system>
class parallel_tempering {
 private:
  std::vector<system *> sys;
  std::vector<double> T;
  std::vector<std::vector<float>> E;
  std::vector<float> E_min;
  std::vector<std::unique_ptr<std::ofstream>> files;
  std::unique_ptr<std::ofstream> file_E_min;
  long steps;
  std::vector<accumulator<double>> num_switch;
  std::mt19937 gen;
  std::uniform_int_distribution<int> uni_int;
  std::uniform_real_distribution<double> uni_double;
  bool print_E;
  bool print_min;
  std::string fname;
  std::unique_ptr<spin_t[]> creat_mask(int i, double rand);
  void new_file(print_select select);
  void old_file(print_select select);
  void open_file(print_select select, bool append);
  void close_file();
 public:
  parallel_tempering(std::vector<system *> &sys_, const std::vector<double> &T_,
                     long steps, std::string &filename);
  void step(int stride);
  void print(print_select select);
  void print_swap(std::ostream &f);
  void set_info(std::string info, print_select select);
  void reset();
  std::vector<float> get_E_min();
  std::ostream &save(std::ostream &stream);
  std::istream &load(std::istream &stream);
  void update_T();
  ~parallel_tempering();
};

template <class system>
parallel_tempering<system>::parallel_tempering(std::vector<system *> &sys_,
                                               const std::vector<double> &T_,
                                               long steps_,
                                               std::string &filename)
    : gen(123), uni_int(0, T_.size() - 2), uni_double(0., 1.) {
  sys = sys_;
  T = T_;
  print_E = false;
  print_min = false;
  fname = filename;
  num_switch.assign(T.size() - 1, {});
  if (T.size() != sys.size()) {
    LOG(LOG_ERROR) << "size difference :";
    LOG(LOG_ERROR) << "sys : " << sys.size();
    LOG(LOG_ERROR) << "T   : " << T.size();
  }
  steps = steps_;
  std::sort(T.begin(), T.end());
  // files.reserve(T.size());
  E.reserve(T.size());
  for (unsigned int i = 0; i < T.size(); ++i) {
    sys[i]->set_T(T[i]);
    E.push_back(std::vector<float>());
  }
}

template <class system>
void parallel_tempering<system>::step(int stride) {
  std::vector<std::vector<float>> *Ep=&E;
  std::vector<system *> *sysp=&sys;
  for (int i = 0; i < steps; ++i) {
    #pragma omp parallel for shared(stride, i,sysp,Ep)
    for (unsigned int j = 0; j < T.size(); ++j) {
      (*sysp)[j]->sweep();


      if (((stride != 0) && ((i % stride) == 0)) || i == (steps - 1)) {
        (*Ep)[j] = (*sysp)[j]->measure();

      }
    }

    if (stride != 0 && (i % stride) == 0) {

      print(PRINT_E);
    } else if (stride == 0) {
      // min E
      if (E_min.empty()) {
        E_min = E[0];
      }
      for ( unsigned int j = 0; j < E_min.size(); ++j) {

        E_min[j] = std::max(E[0][j], E_min[j]);
      }
    }
  }

  int i = uni_int(gen);
  // swap system
  sys[i]->swap(sys[i + 1], creat_mask(i, uni_double(gen)));
}

template <class system>
void parallel_tempering<system>::print(print_select select) {
  new_file(select);
  switch (select) {
    case PRINT_E: {
      for (unsigned int i = 0; i < T.size(); ++i) {
        for (unsigned int j = 0; j < E[i].size(); ++j) {
          (*files[i]) << E[i][j] << "\t";
        }
        (*files[i]) << "\n";
      }
      break;
    }
    case PRINT_MIN: {
      for (unsigned int i = 0; i < E_min.size(); ++i) {
        (*file_E_min) << E_min[i] << "\t";
      }
      (*file_E_min) << std::endl;
      break;
    }
    default: { LOG(LOG_ERROR) << "print select error"; }
  }
}

template <class system>
std::unique_ptr<spin_t[]> parallel_tempering<system>::creat_mask(int i, double rand) {
  std::unique_ptr<spin_t[]> mask(new spin_t[(unsigned int)E[i].size() / 64]);
  for (size_t j = 0; j < (unsigned int)E[i].size() / 64; j++) {
    mask[j] = 0UL;
  }
  int a = 0;
  double dB = (1. / T[i] - 1. / T[i + 1]);
  for (size_t j = 0; j < E[i].size(); ++j) {
    float dE = -(E[i][j] - E[i + 1][j]);
    if (((dE * dB) > 0.) || rand < exp(dB * dE)) {
      mask[j / 64] |= 1UL << j % 64;
      a++;
      LOG(LOG_INFO)<<j<<"\t"<<i<<"\t"<<i+1;
    }
    else{
      LOG(LOG_INFO)<<j<<"\tNaN\tNaN";
    }
  }
  num_switch[i](a / (double)64.);
  return mask;
}

template <class system>
void parallel_tempering<system>::print_swap(std::ostream &f) {
  for (unsigned int i = 0; i < T.size() - 1; ++i) {
    f << T[i] << " -> " << T[i + 1] << "\t: " << num_switch[i].get() << "\n";
  }
}

template <class system>
void parallel_tempering<system>::set_info(std::string info,
                                          print_select select) {
  new_file(select);
  switch (select) {
    case PRINT_E: {
      for (std::unique_ptr<std::ofstream> &x : files) {
        (*x) << info;
      }
      break;
    }
    case PRINT_MIN: {
      (*file_E_min) << info;
      break;
    }
    default: { LOG(LOG_ERROR) << "print select error"; }
  }
}

template <class system>
parallel_tempering<system>::~parallel_tempering() {}

template <class system>
void parallel_tempering<system>::new_file(print_select select) {
  open_file(select, false);
}

template <class system>
void parallel_tempering<system>::old_file(print_select select) {
  open_file(select, true);
}

template <class system>
void parallel_tempering<system>::open_file(print_select select, bool append) {
  std::ios_base::openmode mode = std::ios_base::out;
  if (append) {
    mode = std::ios_base::app;
  }
  switch (select) {
    case PRINT_E: {
      if (!print_E) {
        for (unsigned int i = 0; i < T.size(); ++i) {
          std::string fileT = std::regex_replace(
              fname, std::regex("%", std::regex_constants::grep),
              "T" + std::to_string(T[i]), std::regex_constants::match_default);
          if (fileT == fname) {
            LOG(LOG_ERROR) << "file name hasn't changed, mising % to replase";
          }
          files.push_back(
              std::unique_ptr<std::ofstream>(new std::ofstream(fileT, mode)));
          if (!((files.back())->is_open())) {
            LOG(LOG_ERROR) << "fehler beim ofenen der datei für E";
          }
        }
        print_E = true;
      }
      break;
    }
    case PRINT_MIN: {
      if (!print_min) {
        file_E_min = std::unique_ptr<std::ofstream>(new std::ofstream(
            std::regex_replace(fname,
                               std::regex("%", std::regex_constants::grep),
                               "E_min", std::regex_constants::match_default),
            mode));
        if (!(file_E_min->is_open())) {
          LOG(LOG_ERROR) << "fehler beim ofenen der datei für E_min " << fname
                         << "\t"
                         << std::regex_replace(
                                fname,
                                std::regex("%", std::regex_constants::grep),
                                "E_min", std::regex_constants::match_default);
        }
        print_min = true;
      }
      break;
    }
    default: { LOG(LOG_ERROR) << "print select error"; }
  }
}

template <class system>
void parallel_tempering<system>::close_file() {
  if (print_E) {
    files.clear();
    LOG(LOG_DEBUG) << "close print_E";
  }
  if (print_min) {
    file_E_min->close();
    // delete file_E_min;
    LOG(LOG_DEBUG) << "close print_min";
  }
}

template <class system>
void parallel_tempering<system>::reset() {
  for (size_t i = 0; i < E_min.size(); ++i) {
    E_min[i] = 0.;
  }

  for (system *&x : sys) {
    x->init_rand();
  }

  for (accumulator<double> &x : num_switch) {
    x.reset();
  }
}

template <class system>
std::ostream &parallel_tempering<system>::save(std::ostream &stream) {
  size_t size = 0;
  binary_write(stream, steps, 1);
  binary_write(stream, print_E, 1);
  binary_write(stream, print_min, 1);
  binary_write(stream, gen, 1);
  binary_write(stream, uni_int, 1);
  binary_write(stream, uni_double, 1);

  size = fname.size();
  binary_write(stream, size, 1);
  binary_write_string(stream, fname, size);

  size = T.size();
  binary_write(stream, size, 1);
  binary_write(stream, *(T.data()), size);

  size = E_min.size();
  binary_write(stream, size, 1);
  binary_write(stream, *(E_min.data()), size);

  size = num_switch.size();
  binary_write(stream, size, 1);
  binary_write(stream, *(num_switch.data()), size);

  size = E.size();
  binary_write(stream, size, 1);
  for (size_t i = 0; i < size; i++) {
    size_t size_a = E[i].size();
    binary_write(stream, size_a, 1);
    binary_write(stream, *(E[i].data()), size_a);
    sys[i]->save(stream);
  }
  return stream;
}

template <class system>
std::istream &parallel_tempering<system>::load(std::istream &stream) {
  size_t size = 0;
  close_file();
  binary_read(stream, steps, 1);
  binary_read(stream, print_E, 1);
  binary_read(stream, print_min, 1);
  binary_read(stream, gen, 1);
  binary_read(stream, uni_int, 1);
  binary_read(stream, uni_double, 1);
  binary_read(stream, size, 1);
  fname.resize(size);
  binary_read_string(stream, fname, size);

  binary_read(stream, size, 1);
  double *T_data = new double[size];
  binary_read(stream, *T_data, size);
  T.clear();
  T=std::vector<double>(&T_data[0], &T_data[size]);
  delete[] T_data;

  if (print_E) {
    print_E = false;
    old_file(PRINT_E);
  }
  if (print_min) {
    print_min = false;
    old_file(PRINT_MIN);
  }

  binary_read(stream, size, 1);
  float *E_min_data = new float[size];
  binary_read(stream, *E_min_data, size);
  E_min = std::vector<float>(&E_min_data[0], &E_min_data[size]);
  delete[] E_min_data;

  binary_read(stream, size, 1);
  accumulator<double> *num_switch_data = new accumulator<double>[size];
  binary_read(stream, *num_switch_data, size);
  num_switch.clear();
  num_switch = std::vector<accumulator<double>>(&num_switch_data[0],
                                                &num_switch_data[size]);

  delete[] num_switch_data;

  for (size_t i = 0; i < E.size(); i++) {
    E[i].clear();
  }
  E.clear();
  binary_read(stream, size, 1);
  for (size_t i = 0; i < size; i++) {
    size_t size_a;
    binary_read(stream, size_a, 1);
    float *E_data = new float[size_a];
    binary_read(stream, *E_data, size_a);
    E.push_back(std::vector<float>(&E_data[0], &E_data[size_a]));
    delete[] E_data;
    sys[i]->load(stream);
  }
  return stream;
}

template <class system>
std::vector<float> parallel_tempering<system>::get_E_min() {
  return E_min;
}

template <class system>
void parallel_tempering<system>::update_T(){

  for (size_t i = 0; i < num_switch.size(); i++) {
    if (num_switch[i].get()<0.3) {
      std::ios_base::openmode mode = std::ios_base::out;
      i++;
      num_switch.insert(num_switch.begin()+i,accumulator<double> ());
      T.insert(T.begin()+i,(T[i]+T[i-1])/2.0);
      E.insert(E.begin()+i, std::vector<float>());
      sys.insert(sys.begin()+i,sys[i]->clone());
      sys[i]->set_T(T[i]);

      if(print_E){
        std::string fileT = std::regex_replace(
            fname, std::regex("%", std::regex_constants::grep),
            "T" + std::to_string(T[i]), std::regex_constants::match_default);
        if (fileT == fname) {
          LOG(LOG_ERROR) << "file name hasn't changed, mising % to replase";
        }
        files.insert(files.begin()+i,
            std::unique_ptr<std::ofstream>(new std::ofstream(fileT, mode)));
        if (!((files[i])->is_open())) {
          LOG(LOG_ERROR) << "fehler beim ofenen der datei für E";
        }
      }
      uni_int=std::uniform_int_distribution<int> (uni_int.a(),uni_int.b()+1);
    }
  }
}
