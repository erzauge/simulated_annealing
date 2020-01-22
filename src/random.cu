#include <cuda.h>
#include <curand_kernel.h>
#include "random.cuh"

__global__ void setup_radome(curandStateXORWOW_t *state, unsigned long long seed, unsigned long long N)
{
    unsigned long long id = threadIdx.x+blockDim.x*blockIdx.x ;
    if(id<N){
      /* Each thread gets same seed, a different sequence
         number, no offset */
      curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void setup_radome(curandStatePhilox4_32_10_t *state, unsigned long long seed, unsigned long long N)
{
    unsigned long long id = threadIdx.x+blockDim.x*blockIdx.x;
    if(id<N){
      /* Each thread gets same seed, a different sequence
      number, no offset */
      curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void generate_kernel(curandStatePhilox4_32_10_t *random,unsigned int *buffer,unsigned long long N, int t){
  unsigned long long id = threadIdx.x+blockDim.x*blockIdx.x;
  if(id<N){
    curandStatePhilox4_32_10_t localState = random[id];
    unsigned long long tid=t*id;
    for(int i=0;i<t;i++){
      buffer[tid+i]=curand(&localState);
    }
    random[id] = localState;
  }
}
