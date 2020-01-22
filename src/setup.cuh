#pragma once

#include "Logging.hpp"
#include "stdio.h"

typedef unsigned long long spin_t;

#define CURAND_CALL(x)                                       \
  do {                                                       \
    if ((x) != CURAND_STATUS_SUCCESS) {                      \
      printf("Error at %s:%d\t%i\n", __FILE__, __LINE__, x); \
      return EXIT_FAILURE;                                   \
    }                                                        \
  } while (0)

#define gpuErrchk(ans)                         \
  if ((ans) != cudaSuccess) {                  \
    LOG(LOG_ERROR) << cudaGetErrorString(ans) <<" : "<<ans; \
                                     \
  }

#define BX 16
#define BY 32
#define BL 32
#define BN BL* BL

#define BL_3d 8
#define BN_3d BL_3d* BL_3d* BL_3d

#define SWEEPS_L 1

#define RAND_BUF_SIZE 250000000  // 1GB/64bit

#ifdef DEBUG
#define LastError() gpuErrchk(cudaPeekAtLastError())
#else
#define LastError() ;
#endif
