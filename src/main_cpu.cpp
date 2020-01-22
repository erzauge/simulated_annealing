#include "Logging.hpp"
#include "annealing.hpp"
#include "block.hpp"
#include "cpu_2dp.hpp"
#include "cpu_2d.hpp"
#include "cpu_3d.hpp"
#include "cpu_sys.hpp"
#include "parallel_tempering.hpp"
#include "random_str.hpp"
#include "tools.hpp"
#include "version.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>

#include <tclap/CmdLine.h>

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

int main(int argc, char *argv[]) {

  std::signal(SIGUSR1, signal_callback);
  // configure Logger
  Logger::quiet = false;
  Logger::verbosity = 4;
  Logger::logfilename = "cpulogfile.log";

  // pasing of parameter
  vector<std::string> allowedDim;
  allowedDim.push_back("2p");
  allowedDim.push_back("3");
  allowedDim.push_back("2");
  TCLAP::ValuesConstraint<std::string> allowedDimVals(allowedDim);
  std::vector<TCLAP::Arg *> xorlist;

  TCLAP::CmdLine cmd("", ' ', VERSION, true);

  TCLAP::ValueArg<int> stepsArg("s", "steps", "Number of cooling steps", true,
                                1000, "int");
  cmd.add(stepsArg);
  TCLAP::ValueArg<double> aArg("a", "a", "cooling factor", false, 0.9,
                               "double");
  cmd.add(aArg);
  TCLAP::ValueArg<int> sweepsArg(
      "S", "sweeps", " Number of sweeps per cooling steps", false, 1, "int");
  cmd.add(sweepsArg);
  TCLAP::ValueArg<int> LArg("L", "size", "sys size", false, 64, "int");
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
  cmd.add(reload);
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
                                 &allowedDimVals);
  cmd.add(DimArg);
  TCLAP::SwitchArg timeing("t", "time", "measure speed of spin flip", cmd,
                           false);
  TCLAP::SwitchArg image(
      "i", "image", "save images of sys from the last sweep", cmd, false);
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

  LOG(LOG_DEBUG)<<info;

  info += "\n# Version : ";
  info += VERSION;
  info += "\n";

  // set verbosity
  Logger::verbosity = verbosityArg.getValue();

  string rand_id = random_string(5);

  if (timeing.getValue()) {
    cpu_sys *sys2;

    if (DimArg.getValue() == "2p") {
      sys2 = new cpu_2dp(LArg.getValue(), TArg.getValue(),
                           hArg.getValue());  //,sys_2(L,T,h);
    } else if (DimArg.getValue() == "3") {
      sys2 = new cpu_3d(LArg.getValue(), TArg.getValue(), hArg.getValue());
    } else if (DimArg.getValue() == "2") {
      sys2 = new cpu_2d(LArg.getValue(), TArg.getValue(), hArg.getValue());
    } else {
      sys2 = NULL;
      LOG(LOG_ERROR) << "error in selecting sys";
    }


    sys2->init_J();
    sys2->init_rand();

    double t, t2, t_min, t_max;
    t = 0;
    t2 = 0;
    t_min = NAN;
    t_max = NAN;

    long steps = stepsArg.getValue();

    for (int i = 0; i < steps; ++i) {

      auto start = std::chrono::high_resolution_clock::now();
      sys2->sweep();
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> diff = end - start;

      t += diff.count();
      t2 += diff.count() * diff.count();
      t_min = min(diff.count(), t_min);
      t_max = max(diff.count(), t_max);
    }

    LOG(LOG_INFO) << VERSION;
    FILE *fp = popen(
        "cat /proc/cpuinfo| grep \"model name\"|head -n1|cut -d ':' -f 2|tr -d "
        "'\n'",
        "r");
    string name;
    char buffer[10];
    while (fgets(buffer, 10, fp) != NULL) {
      name += string(buffer);
    }
    pclose(fp);

    LOG(LOG_INFO) << name;

    LOG(LOG_INFO) << DimArg.getValue() << "\t" << LArg.getValue() << "\t"
                  << steps;

    LOG(LOG_INFO) << (t / (steps)) / (sys2->get_N() * 64) << "\t"
                  << sqrt(t2 / (steps)-t / (steps)*t / (steps)) /
                         (sys2->get_N() * 64)
                  << "\t" << (t_min) / (sys2->get_N() * 64) << "\t"
                  << (t_max) / (sys2->get_N() * 64);


    delete sys2;

    return 0;
  }

  // set output stream
  ostream *fp = &cout;
  ofstream fout;
  if (outArg.isSet()) {
    fout.open(outArg.getValue());
    if (fout.is_open()) {
      fp = &fout;
    } else {
      LOG(LOG_ERROR) << "feher beim öfen des output file :"
                     << outArg.getValue();
      exit(-1);
    }
  }

  // preoer tpertues for paralletempering
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
  std::vector<cpu_sys *> sys;
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
      sys.push_back(new cpu_2dp(LArg.getValue(), TArg.getValue(),
                                  hArg.getValue()));
    } else if (DimArg.getValue() == "3") {
      sys.push_back(
          new cpu_3d(LArg.getValue(), TArg.getValue(), hArg.getValue()));
    } else if (DimArg.getValue() == "2") {
      sys.push_back(new cpu_2d(LArg.getValue(), TArg.getValue(),
                                  hArg.getValue()));
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
    annealing<cpu_sys> annealer(TArg.getValue(), sweepsArg.getValue(),
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

    parallel_tempering<cpu_sys> tempering(sys, T, sweepsArg.getValue(),
                                          outTArg.getValue());

    tempering.set_info(info, repeatArg.isSet() ? PRINT_MIN : PRINT_E);
    std::vector<float> *E_min = new std::vector<float>[repeatArg.getValue()];
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

          E_min[i] = tempering.get_E_min();
          tempering.update_T();
          tempering.save(otemp);
          otemp.close();
          tempering.reset();

        }
      }
      if (repeatArg.isSet()&&(repeatArg.getValue()!=1)) {

        first = false;
        std::vector<int> c = max_count(E_min, repeatArg.getValue());

        found = all_grater(c, repeatArg.getValue() / 2);

      } else {
        found = true;
      }

      if(max_iter.isSet()&&count>max_iter.getValue()){
        found = true;
      }

      if (found) {
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

  // sav  e sys image
  if (image.isSet()) {
    for   (int i = 0; i < num; ++i) {
      sys[i]->save_sys("cpu_sys_T"+std::to_string(sys[i]->get_T())+"_");
    }
  }

  // cleanup
  for (int i = 0; i < num; ++i) {
    delete sys[i];
  }

  return 0;
}
