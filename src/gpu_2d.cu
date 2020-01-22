#include "gpu_2d.cuh"
#include "device_2d.cuh"
#include "setup.cuh"
#include "bin_io.hpp"
#include "sys_file.hpp"
#include "Logging.hpp"
#include "random.cuh"

#include <curand.h>
#include <curand_kernel.h>


#include <cmath>
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

gpu_2d::gpu_2d(int L_, double T_, double h_){

	L=L_;
	N=L*L;

	T=T_;
	h=h_;


	curandCreateGenerator(&gen2, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);

	//allocate memory on divice

	cudaMalloc(&gen_d,N*sizeof(curandStatePhilox4_32_10_t));
	cudaMalloc(&s1_D, N*sizeof(spin_t));
	cudaMalloc(&s2_D, N*sizeof(spin_t));
	cudaMalloc(&J_x_D, 2*N*sizeof(int2));
	cudaMalloc(&J_y_D, 2*N*sizeof(int2));
	cudaMalloc(&boltz_D, 20*sizeof(float));
	cudaMalloc(&M_buf_D, 2*64*ceil(N/256.+1)*sizeof(float));
	cudaMalloc(&EJ_buf_D, 2*64*ceil(N/256.+1)*sizeof(float));

	//init randome

	setup_radome<<<ceil(N/256.),256>>>(gen_d,1234ULL,N);



	//set boltzmanfactor
	float boltz_H[20]; //boltzmanfaktor Host

	#ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz_H[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz_H[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz_H[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz_H[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  #else
  for (int i = 0; i < 4; ++i) {
    int j = 1 + i;
    boltz_H[i] = exp(-2. * j * (1. / T));
  }
  boltz_H[4] = exp(-2. * h / T);
  #endif




	//loade and binde texure memory
	cudaMemcpy(boltz_D, boltz_H, 20*sizeof(float), cudaMemcpyHostToDevice);
	/

	//setup strems

	for (int i = 0; i < 4; ++i)
	{
		cudaStreamCreate(&stream[i]);
	}
	LastError();
}

gpu_2d::~gpu_2d(){

}

void gpu_2d::gpu_free(){
	for (int i = 0; i < 4; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}
	cudaFree(boltz_D);
	cudaFree(s1_D);
	cudaFree(s2_D);
	cudaFree(J_x_D);
	cudaFree(J_y_D);

	cudaFree(M_buf_D);
	cudaFree(EJ_buf_D);

	// curandDestroyGenerator(gen);
	curandDestroyGenerator(gen2);
	LastError();
}

void gpu_2d::sweep(){
	LastError();

	dim3 block(16,16);
 	dim3 grid(ceil(L/16.),ceil(L/16.));

	metrpolis_2d<<<grid,block,0,stream[1]>>>(s1_D,s2_D,gen_d,boltz_D,L,0);
	metrpolis_2d<<<grid,block,0,stream[1]>>>(s2_D,s1_D,gen_d,boltz_D,L,0);


	LastError();
}

void gpu_2d::set_T(double T_){
	T=T_;
	float boltz_H[20]; //boltzmanfaktor Host

	#ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz_H[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz_H[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz_H[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz_H[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  #else
  for (int i = 0; i < 4; ++i) {
    int j = 1 + i;
    boltz_H[i] = exp(-2. * j * (1. / T));
  }
  boltz_H[4] = exp(-2. * h / T);
  #endif

	cudaDeviceSynchronize();

	cudaMemcpy(boltz_D, boltz_H, 20*sizeof(float), cudaMemcpyHostToDevice);

	LastError();
}

void gpu_2d::set_h(double h_){
	h=h_;
	float boltz_H[20]; //boltzmanfaktor Host

	#ifndef SPLIT_H
  for (size_t i = 0; i < 5; i++) {
    int j=(i-2)*2;
    boltz_H[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz_H[5+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
    boltz_H[10+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1-h)));
    boltz_H[15+i] = std::min(1., (i==0)?0.:exp(-2*(1/T)*(j-1+h)));
  }
  #else
  for (int i = 0; i < 4; ++i) {
    int j = 1 + i;
    boltz_H[i] = exp(-2. * j * (1. / T));
  }
  boltz_H[4] = exp(-2. * h / T);
  #endif
	cudaDeviceSynchronize();

	cudaMemcpy(boltz_D, boltz_H, 20*sizeof(float), cudaMemcpyHostToDevice);

	LastError();
}


vector<float> gpu_2d::measure(){
	int c=(int)ceil(N/(256.*2));
	float *M_H=new float[2*64*c];
	float *EJ_H=new float[2*64*c];
	LastError();
	dim3 grid(ceil(N/(2*256.)),1,1);
	dim3 block(256,1,1);
	dim3 grid_s(ceil(L/16.),ceil(L/16.));
	dim3 block_s(16,16);
	checkerbord_switch_2d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L);
	measure_EJ_M_2d<<<grid,block,0,stream[1]>>>(s1_D,&EJ_buf_D[0],&M_buf_D[0],N,L,0);
	cudaMemcpy(&M_H[0],&M_buf_D[0],c*64*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&EJ_H[0],&EJ_buf_D[0],c*64*sizeof(float),cudaMemcpyDeviceToHost);
	measure_EJ_M_2d<<<grid,block,0,stream[2]>>>(s2_D,&EJ_buf_D[64*c],&M_buf_D[64*c],N,L,0);
	cudaMemcpy(&M_H[c*64],&M_buf_D[c*64],c*64*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&EJ_H[c*64],&EJ_buf_D[c*64],c*64*sizeof(float),cudaMemcpyDeviceToHost);
	LastError();
	//cudaDeviceSynchronize();
	checkerbord_switch_2d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L);

	vector<float> result;
	result.assign(2*64, 0);
	for (int i = 0; i < 2*64; ++i)
	{
		result[i]=-L;//-L als korektur von rand beningungen
		for (int j = 0; j < c;++j)
		{
			result[i]+=EJ_H[i*c+j]/*/N*/+h*M_H[i*c+j]/*/N*/;
		}
	}
	delete[] M_H;
	delete[] EJ_H;


	return result;
}

void gpu_2d::load_J(string fname){
	spin_t *Jx =new spin_t[2*N];
	spin_t *Jy =new spin_t[2*N];
	int error=load_J_2d(Jx,Jy,N,fname);
	if(error<0){
		LOG(LOG_WARNING)<<"fehler beim laden von J in file \""<<fname<<"\" mit fehler "<<error;
		return;
	}
	cudaDeviceSynchronize();
	cudaMemcpy(J_x_D, Jx, 2*N*sizeof(spin_t), cudaMemcpyHostToDevice);
	cudaMemcpy(J_y_D, Jy, 2*N*sizeof(spin_t), cudaMemcpyHostToDevice);
	cudaBindTexture(0, get_J_yi(), J_y_D,2*N*sizeof(int2));
	cudaBindTexture(0, get_J_xi(), J_x_D,2*N*sizeof(int2));
	delete[] Jx;
	delete[] Jy;
	LastError();
}

void gpu_2d::save_J(string fname){
	spin_t *Jx =new spin_t[2*N];
	spin_t *Jy =new spin_t[2*N];
	cudaDeviceSynchronize();
	cudaMemcpy(Jx, J_x_D, 2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
	cudaMemcpy(Jy, J_y_D, 2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
       	int error=save_J_2d(Jx,Jy,N,fname);
	if(error<0){
		LOG(LOG_WARNING)<<"fehler beim laden von J in file \""<<fname<<"\" mit fehler "<<error;
	}
	delete[] Jx;
	delete[] Jy;
	LastError();
}


void gpu_2d::init_J(){
	unsigned int *buffer_D; //Buffer divice
	cudaMalloc(&buffer_D, 2*2*N*sizeof(unsigned int));
	//ranomly initlize J
	generate_kernel<<<ceil(N/256.),256>>>(gen_d,buffer_D,N, 2*2);
	J_order<<<ceil(N/256.),256>>>(J_x_D,J_y_D,buffer_D,L,N);
	cudaFree(buffer_D);
	cudaBindTexture(0, get_J_yi(), J_y_D,2*N*sizeof(int2));
	cudaBindTexture(0, get_J_xi(), J_x_D,2*N*sizeof(int2));
	LastError();
}

void gpu_2d::init_rand(){
	//ranomly initlize spins
	curandGenerateLongLong(gen2,s1_D,N);
	curandGenerateLongLong(gen2,s2_D,N);
	LastError();
}

void gpu_2d::set_seed(long long seed_){
	// curandSetPseudoRandomGeneratorSeed(gen, seed_);
	curandSetPseudoRandomGeneratorSeed(gen2, seed_);

	setup_radome<<<ceil(N/256.),256>>>(gen_d,1234ULL,N);

	LastError();
}

void gpu_2d::save_sys(string prefix){
	spin_t s[N];
	cudaDeviceSynchronize();
	cudaMemcpy(s,s1_D,N*sizeof(spin_t),cudaMemcpyDeviceToHost);

	for (int i = 0; i < 64; ++i)
	{
		// image setup
		stringstream convert;
		convert<<prefix<<i<<".pbm";
		ofstream file(convert.str().c_str());
		file<<"P1"<<endl;
		file<<L<<" "<<L<<endl;
		// print image
		for (int j = 0; j < L; ++j)
		{
			for (int k = 0; k < L; ++k)
			{
				file<<((s[j*L+k]&((spin_t)1<<i))==0?"0 ":"1 ");
			}
			file<<endl;
		}
	}
	LastError();
}

long gpu_2d::get_N(){
	return N;
}

void gpu_2d::swap(gpu_sys *sys, std::unique_ptr<spin_t[]> mask) {
  gpu_2d *sys_2d = dynamic_cast<gpu_2d *>(sys);
  if (sys_2d != NULL) {
    swap(sys_2d, move(mask));
  } else {
    LOG(LOG_ERROR) << "conversion error";
  }
}

void gpu_2d::swap(gpu_2d *sys, std::unique_ptr<spin_t[]> mask) {
  // LOG(LOG_INFO)<<sys->N;
	dim3 grid_s(ceil(L/16.),ceil(L/16.));
	dim3 block_s(16,16);
	checkerbord_switch_2d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L);
	checkerbord_switch_2d<<<grid_s,block_s,0,stream[1]>>>(sys->s1_D,sys->s2_D,sys->L);
	swap_2d<<<ceil(N/256.),256,0,stream[1]>>>(s1_D,sys->s1_D,mask[0],N);
	swap_2d<<<ceil(N/256.),256,0,stream[1]>>>(s2_D,sys->s2_D,mask[1],N);
	checkerbord_switch_2d<<<grid_s,block_s,0,stream[1]>>>(sys->s1_D,sys->s2_D,sys->L);
	checkerbord_switch_2d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L);
}


ostream & gpu_2d::save(ostream &stream) {
	LastError();
  binary_write(stream, L, 1);
  binary_write(stream, T, 1);
  binary_write(stream, h, 1);
	// spins
	LastError();
	cudaDeviceSynchronize();
	spin_t *s=new spin_t[N];
	cudaMemcpy(s,s1_D,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
  binary_write(stream, s[0], N);
	cudaMemcpy(s,s2_D,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
	binary_write(stream, s[0], N);
	delete[] s;
	// copling
	LastError();
	spin_t *J=new spin_t[2*N];
	cudaMemcpy(J,J_x_D,2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
  binary_write(stream, J[0], 2 * N);//x
	cudaMemcpy(J,J_y_D,2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
  binary_write(stream, J[0], 2 * N);//y
	delete [] J;
	// RNG state
	LastError();
	curandStatePhilox4_32_10_t *gen_n_save=new curandStatePhilox4_32_10_t[N];
	cudaMemcpy(gen_n_save,gen_d,N*sizeof(curandStatePhilox4_32_10_t),cudaMemcpyDeviceToHost);
	binary_write(stream, gen_n_save[0], N);
	delete[] gen_n_save;
	LastError();
  return stream;
}

istream & gpu_2d::load(istream &stream){
	LastError();
	binary_read(stream, L, 1);
  N = L * L;
  binary_read(stream, T, 1);
  set_T(T);
  binary_read(stream, h, 1);
  set_h(h);
	cudaDeviceSynchronize();

	// memory
	cudaFree(gen_d);
	cudaFree(s1_D);
	cudaFree(s2_D);
	cudaFree(J_x_D);
	cudaFree(J_y_D);
	cudaFree(M_buf_D);
	cudaFree(EJ_buf_D);
	cudaMalloc(&gen_d,N*sizeof(curandStatePhilox4_32_10_t));
	cudaMalloc(&s1_D, N*sizeof(spin_t));
	cudaMalloc(&s2_D, N*sizeof(spin_t));
	cudaMalloc(&J_x_D, 2*N*sizeof(int2));
	cudaMalloc(&J_y_D, 2*N*sizeof(int2));
	cudaMalloc(&M_buf_D, 2*64*ceil(N/256.+1)*sizeof(float));
	cudaMalloc(&EJ_buf_D, 2*64*ceil(N/256.+1)*sizeof(float));

	// spins
	spin_t *s=new spin_t[N];
	binary_read(stream, s[0], N);
	cudaMemcpy(s1_D,s,N*sizeof(spin_t),cudaMemcpyHostToDevice);
	binary_read(stream, s[0], N);
	cudaMemcpy(s2_D,s,N*sizeof(spin_t),cudaMemcpyHostToDevice);
	delete[] s;

	// copling
	spin_t *J=new spin_t[2*N];
	binary_read(stream, J[0], 2 * N);//x
	cudaMemcpy(J_x_D,J,2*N*sizeof(spin_t),cudaMemcpyHostToDevice);
	binary_read(stream, J[0], 2 * N);//y
	cudaMemcpy(J_y_D,J,2*N*sizeof(spin_t),cudaMemcpyHostToDevice);
	delete[] J;
	cudaBindTexture(0, get_J_yi(), J_y_D,2*N*sizeof(int2));
	cudaBindTexture(0, get_J_xi(), J_x_D,2*N*sizeof(int2));
	// RNG state
	curandStatePhilox4_32_10_t *gen_n_save=new curandStatePhilox4_32_10_t[N];
	binary_read(stream, gen_n_save[0], N);
	cudaMemcpy(gen_d,gen_n_save,N*sizeof(curandStatePhilox4_32_10_t),cudaMemcpyHostToDevice);
	delete[] gen_n_save;

	LastError();
	cudaDeviceSynchronize();

	return stream;
}
