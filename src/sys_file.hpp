#pragma once

#include <string>
using namespace std;

int load_J_2d(unsigned long long *Jx, unsigned long long *Jy, long N, string fname);

int save_J_2d(unsigned long long *Jx, unsigned long long *Jy, long N, string fname);

int load_J_3d(unsigned long long *Jx, unsigned long long *Jy,unsigned long long *Jz,long N, string fname);

int save_J_3d(unsigned long long *Jx,unsigned long long *Jy,unsigned long long *Jz,long N, string fname);