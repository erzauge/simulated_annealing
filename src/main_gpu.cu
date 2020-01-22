#include "Logging.hpp"
#include "annealing.hpp"
#include "block.hpp"
#include "gpu_2d.cuh"
#include "gpu_3d.cuh"
#include "parallel_tempering.hpp"
#include "random_str.hpp"
#include "tools.hpp"
#include "version.h"

#include <cmath>
#include <csignal>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include <string>

#include <tclap/CmdLine.h>

using namespace std;

#ifndef VERSION
#define VERSION "empty"
#endif

namespace TCLAP {
template <>
struct ArgTraits<block> {
  typedef ValueLike ValueCategory;
};
}  // namespace TCLAP

volatile std::sig_atomic_t URS1_Status = 0;

void signal_callback(int signal) {
  if (signal == SIGUSR1) {
    URS1_Status = 1;
  }
}

int main(int argc, char const *argv[]) {
  std::signal(SIGUSR1, signal_callback);
  // configure Logger
  Logger::quiet = false;
  Logger::verbosity = 4;
  Logger::logfilename = "gpulogfile.log";

  // pasing of parameter
  vector<std::string> allowed;
  allowed.push_back("2p");
  allowed.push_back("3");
  TCLAP::ValuesConstraint<std::string> allowedVals(allowed);
  std::vector<TCLAP::Arg *> xorlist;
  TCLAP::CmdLine cmd("", ' ', VERSION, true);

  TCLAP::ValueArg<int> stepsArg("s", "steps", "Number of cooling steps", false,
                                100, "int");
  cmd.add(stepsArg);
  TCLAP::ValueArg<double> aArg("a", "a", "cooling factor", false, 0.9,
                               "double");
  cmd.add(aArg);
  TCLAP::ValueArg<int> sweepsArg(
      "S", "sweeps", " Number of sweeps per cooling steps", false, 100, "int");
  cmd.add(sweepsArg);
  TCLAP::ValueArg<int> LArg("L", "size", "system size", false, 64, "int");
  cmd.add(LArg);
  TCLAP::ValueArg<double> TArg("T", "T", "starting Temperatur", false, 2.26,
                               "double");
 xorlist.push_back(&TArg);
 TCLAP::MultiArg<double> PArg(
     "P", "parallel_tempering",
     "parallel tempering Temperaturs. needs to be called multipel times",
     false, "double");
 xorlist.push_back(&PArg);
 TCLAP::MultiArg<block> PbArg("", "Pb",
                              "parallel tempering Temperaturs blocks ", false,
                              "start step stop (double)");
 xorlist.push_back(&PbArg);
 TCLAP::ValueArg<string> P_fileArg(
     "", "P_file", "file of Temperaturs blocks for parallel tempering", false,
     "", "string");
 xorlist.push_back(&P_fileArg);
 TCLAP::ValueArg<string> reload("", "reload", "reload stades frome file",
                                false, "", "string");
 xorlist.push_back(&reload);
 cmd.xorAdd(xorlist);
  TCLAP::ValueArg<double> hArg("m", "h", "magnetic filed", false, 0., "double");
  cmd.add(hArg);
  TCLAP::ValueArg<string> outArg("o", "output", "output file name", false,
                                 "cout", "string");
  cmd.add(outArg);
  TCLAP::ValueArg<string> outTArg("", "outputT",
                                  "output file name for parallel tempering",
                                  false, "out_%.dat", "string");
  cmd.add(outTArg);
  TCLAP::ValueArg<long> seedArg("", "seed", "seed for RNG", false, 1234,
                                "long");
  cmd.add(seedArg);
  TCLAP::ValueArg<string> load_JArg("j", "load_J", "load J file name", false,
                                    "", "string");
  cmd.add(load_JArg);
  TCLAP::ValueArg<string> save_JArg("J", "save_J", "save J file name", false,
                                    "", "string");
  cmd.add(save_JArg);
  TCLAP::ValueArg<string> DimArg("d", "dim", "select dim", false, "2p",
                                 &allowedVals);
  cmd.add(DimArg);
  TCLAP::SwitchArg timeing("t", "timing", "measure speed of spin flip", cmd,
                           false);
  TCLAP::SwitchArg image("i", "image", "save images of system", cmd, false);
  TCLAP::SwitchArg swapFlag("", "swap",
                            "prints swap properbeletie for parallel tempering",
                            cmd, false);
  TCLAP::ValueArg<long> verbosityArg("v", "verbosity", "verbosity level", false,
                                     4, "unsigned int");
  cmd.add(verbosityArg);
  TCLAP::ValueArg<int> repeatArg(
      "r", "repeat", "how often to restart the annealing", false, 1, "int");
  cmd.add(repeatArg);
  TCLAP::ValueArg<string> tmp_path("", "tmp", "path to temporay directory",
                                   false, "/tmp/", "string");
  cmd.add(tmp_path);

  TCLAP::ValueArg<string> save_path("", "home", "path to directory where to save recuvery files",
                                   false, ".", "string");
  cmd.add(save_path);

  TCLAP::ValueArg<int> max_iter("", "maxiter", "mx nuber of iteraions",
                                   false, 10, "int");
  cmd.add(max_iter);
  cmd.parse(argc, argv);

  // creat info string
  string info;
  info = "#";
  for (int i = 0; i < argc; ++i) {
    info += " ";
    info += argv[i];
  }

  info += "\n# Version : ";
  info += VERSION;
  info += "\n";

  // set verbosity
  Logger::verbosity = verbosityArg.getValue();

  // get randome string
  string rand_id = random_string(5);

  // timing
  if (timeing.getValue()) {
    gpu_sys *system2;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (DimArg.getValue() == "2p") {
      system2 = new gpu_2d(LArg.getValue(), TArg.getValue(),
                           hArg.getValue());  //,system_2(L,T,h);
    } else if (DimArg.getValue() == "3") {
      system2 = new gpu_3d(LArg.getValue(), TArg.getValue(), hArg.getValue());
    }

    system2->init_J();
    system2->init_rand();

    LastError();
    double t, t2, t_min, t_max;
    t = 0;
    t2 = 0;
    t_min = NAN;
    t_max = NAN;

    long steps = stepsArg.getValue();

    for (int i = 0; i < steps; ++i) {
      cudaEventRecord(start);
      system2->sweep();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);
      t += milliseconds;
      t2 += milliseconds * milliseconds;
      t_min = min(t_min, milliseconds);
      t_max = max(t_max, milliseconds);
    }

    LOG(LOG_INFO) << VERSION;
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    LOG(LOG_INFO) << prop.name;

    LOG(LOG_INFO) << DimArg.getValue() << "\t" << LArg.getValue() << "\t"
                  << steps;
    double flips_per_sweep = system2->get_N() * 64 * 2;
    LOG(LOG_INFO) << (t / (steps * 1000 * flips_per_sweep)) << "\t"
                  << sqrt(t2 / (steps * 1000 * flips_per_sweep) -
                          t / (steps * 1000 * flips_per_sweep) * t /
                              (steps * 1000 * flips_per_sweep))
                  << "\t" << (t_min) / (1000 * flips_per_sweep) << "\t"
                  << (t_max) / (1000 * flips_per_sweep);

    system2->gpu_free();
    exit(0);
  }

  // set output stream
  ostream *fp = &cout;
  ofstream fout;
  if (outArg.isSet()) {
    fout.open(outArg.getValue());
    if (fout.is_open()) {
      fp = &fout;
    } else {
      LOG(LOG_ERROR) << "feher beim öfen des output files" << endl;
      exit(-1);
    }
  }

  /// preoer tpertues for paralletempering
  std::vector<block> T_blocks;
  if (P_fileArg.isSet()) {
    block b;
    ifstream file(P_fileArg.getValue());
    if (!file.is_open()) {
      LOG(LOG_ERROR) << "feher beim öfen des output file :"
                     << P_fileArg.getValue();
      exit(-1);
    }
    while (file >> b) {
      T_blocks.push_back(b);
    }
  } else if (PbArg.isSet()) {
    T_blocks = PbArg.getValue();
  }
  // inizalise sys and selct type
  std::vector<gpu_sys *> sys;
  int num = 1;
  if (PArg.isSet()) {
    num = PArg.getValue().size();
  } else if (PbArg.isSet() || P_fileArg.isSet()) {
    num = 0;
    for (unsigned int i = 0; i < T_blocks.size(); ++i) {
      num += T_blocks[i].size();
    }
  }
  for (int i = 0; i < num; ++i) {
    if (DimArg.getValue() == "2p") {
      sys.push_back(new gpu_2d(LArg.getValue(), TArg.getValue(),
                                  hArg.getValue()));  //,sys_2(L,T,h);
    } else if (DimArg.getValue() == "3") {
      sys.push_back(
          new gpu_3d(LArg.getValue(), TArg.getValue(), hArg.getValue()));
    } else {
      sys.push_back(NULL);
      LOG(LOG_ERROR) << "error in selecting sys";
    }
    // creas or load J
    if (load_JArg.isSet()) {
      sys[i]->load_J(load_JArg.getValue());
    } else {
      sys[i]->set_seed(seedArg.getValue());
      sys[i]->init_J();
      if (save_JArg.isSet()) {
        sys[i]->save_J(save_JArg.getValue());
      }
    }

    // set seed and initialize sys
    sys[i]->set_seed(seedArg.getValue());

    sys[i]->init_rand();
  }
  // selcte paralle tempering or annealing
  if (TArg.isSet()) {
    // initialize annealer
    annealing<gpu_sys> annealer(TArg.getValue(), sweepsArg.getValue(),
                                aArg.getValue(), sys[0], *fp);
    (*fp) << info;
    // set printing flags for annealing
    print_select select = PRINT_E;
    if (repeatArg.isSet()) {
      select = PRINT_MIN;
    }

    // annealing proses
    for (int j = 0; j < repeatArg.getValue(); ++j) {
      for (int i = 0; i < stepsArg.getValue(); ++i) {
        annealer.step(1, select);
      }
      if (repeatArg.isSet()) {
        annealer.print(select);
        annealer.set_T(TArg.getValue());
        sys[0]->init_rand();
      }
    }
  } else {
    // paralle tempering
    if (swapFlag.isSet()) {
      (*fp) << info;
    }
    std::vector<double> T;
    if (PArg.isSet()) {
      if (PArg.getValue().size() < 2) {
        LOG(LOG_ERROR) << "not enoug temperatures. curently :"
                       << PArg.getValue().size();
        exit(-1);
      }
      T = PArg.getValue();
    } else if (PbArg.isSet() || P_fileArg.isSet()) {
      for (unsigned int i = 0; i < T_blocks.size(); ++i) {
        for (int j = 0; j < T_blocks[i].size(); ++j) {
          T.push_back(T_blocks[i][j]);
        }
      }
    }

    parallel_tempering<gpu_sys> tempering(sys, T, sweepsArg.getValue(),
                                          outTArg.getValue());
    tempering.set_info(info, repeatArg.isSet() ? PRINT_MIN : PRINT_E);
    std::vector<float> *E_min = new std::vector<float>[2*repeatArg.getValue()];
    bool found = false;
    bool first = true;
    int count=-1;
    while (!found) {
      count++;
      if (URS1_Status == 1) {
        std::string command = "cp " + tmp_path.getValue() + rand_id + "_* "+save_path.getValue();
        system(command.c_str());
        for (int i = 0; i < argc; ++i) {
          std::cout << " " << argv[i];
        }
        std::cout << " --reload " << save_path.getValue()+rand_id << std::endl;
	      exit(0);
      }

      for (int i = 0; i < repeatArg.getValue(); ++i) {

        if (!first) {  // wen nicht erstern duchlauf
          ifstream itemp(tmp_path.getValue() + rand_id + "_" +
                             std::to_string(i) + ".tempsys",
                         ios::binary);
          if(!itemp.good()) {
            LOG(LOG_ERROR)<<"error by opening file "<< tmp_path.getValue() + rand_id + "_" +
                             std::to_string(i) + ".tempsys";
          }
          tempering.load(itemp);
          itemp.close();
        } else {
          if (reload.isSet()) {
            ifstream itemp(
                reload.getValue() + "_" + std::to_string(i) + ".tempsys",
                ios::binary);
            tempering.load(itemp);
            itemp.close();
          }
        }

        for (int j = 0; j < stepsArg.getValue(); ++j) {
          tempering.step(repeatArg.isSet() ? 0 : 1);
        }

        if (swapFlag.isSet()) {
          tempering.print_swap(*fp);
        }
        if (repeatArg.isSet()) {

          ofstream otemp(tmp_path.getValue() + rand_id + "_" +
                             std::to_string(i) + ".tempsys",
                         ios::binary|ios::out|ios::trunc);
          if(!otemp.good()){
            LOG(LOG_ERROR)<<"error by opening file "<<tmp_path.getValue() + rand_id + "_" +
                             std::to_string(i) + ".tempsys";
          }
          std::vector<float> E_buffer(tempering.get_E_min());
          E_min[2*i] = std::vector<float> (&E_buffer[0],&E_buffer[64]);
          E_min[2*i+1] = std::vector<float> (&E_buffer[64],&E_buffer[128]);
          tempering.save(otemp);
          otemp.close();
          tempering.reset();
        }
      }
      if (repeatArg.isSet()&&(repeatArg.getValue()!=1)) {
        first = false;
        std::vector<int> c = max_count(E_min, 2*repeatArg.getValue());

        found = all_grater(c, repeatArg.getValue() );
      } else {
        found = true;
      }
      if(max_iter.isSet()&&count>max_iter.getValue()){
        found = true;
      }
      if (found) { //if "ground state" found

        for (int i = 0; i < repeatArg.getValue(); i++) {
          ifstream itemp(tmp_path.getValue() + rand_id + "_" +
                             std::to_string(i) + ".tempsys",
                         ios::binary);
          tempering.load(itemp);
          itemp.close();

          if(repeatArg.isSet()){
            tempering.print(PRINT_MIN);
          }
          else{
            tempering.print(PRINT_E);
          }
        }
      }
    }
  }

  // save system image
  if (image.isSet()) {
    for   (int i = 0; i < num; ++i) {
      sys[i]->save_sys("gpu_sys_");
    }
  }

  // cleanup
  for (int i = 0; i < num; ++i) {
    sys[i]->gpu_free();
    delete sys[i];
  }

  // cleanup
  cudaProfilerStop();
  return 0;
}
