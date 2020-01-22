#include "gpu_3d.cuh"
#include "device_3d.cuh"
#include "setup.cuh"
#include "bin_io.hpp"
#include "sys_file.hpp"
#include "Logging.hpp"
#include "random.cuh"

#include <curand.h>
// #include <cub/cub.cuh>

#include <cmath>
#include <stdio.h>

#include <iostream>
#include <sstream>



gpu_3d::gpu_3d(int L_, double T_, double h_){
	L=L_;
	N=L*L*L;

	T=T_;
	h=h_;

	//init random gen
	curandCreateGenerator(&gen2, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);

	//allocate memory on divice

	cudaMalloc(&gen_d,N*sizeof(curandStatePhilox4_32_10_t));
	gpuErrchk(cudaMalloc(&s1_D, N*sizeof(spin_t)));
	gpuErrchk(cudaMalloc(&s2_D, N*sizeof(spin_t)));
	gpuErrchk(cudaMalloc(&J_x_D, 2*N*sizeof(int2)));
	gpuErrchk(cudaMalloc(&J_y_D, 2*N*sizeof(int2)));
	gpuErrchk(cudaMalloc(&J_z_D, 2*N*sizeof(int2)));
	gpuErrchk(cudaMalloc(&boltz_D, 14*sizeof(float)));

	gpuErrchk(cudaMalloc(&M_buf_D, 2*64*ceil(N/256.+1)*sizeof(float)));
	gpuErrchk(cudaMalloc(&EJ_buf_D, 2*64*ceil(N/256.+1)*sizeof(float)));

	// cudaFree(buffer_D);
	setup_radome<<<ceil(N/256.),256>>>(gen_d,1234ULL,N);

	curandGenerateLongLong(gen2,s1_D,N);
	curandGenerateLongLong(gen2,s2_D,N);

	//set boltzmanfactor
	float boltz_H[14]; //boltzmanfaktor Host
	#ifndef SPLIT_H
	for (size_t i = 0; i < 7; i++) {
    int j=(i-3)*2;
    boltz_H[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz_H[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
  }
	#else
	for (int i = 0; i < 3; ++i)
	{
		int j=2+2*i;
		boltz_H[i] = exp(-2.*j*(1./T));
	}
	boltz_H[3]=exp(-2*h/T);
	#endif

	//loade and binde texure memory
	cudaMemcpy(boltz_D, boltz_H, 14*sizeof(float), cudaMemcpyHostToDevice);


	//setup strems

	for (int i = 0; i < 4; ++i)
	{
		cudaStreamCreate(&stream[i]);
	}

}


void gpu_3d::gpu_free(){
	for (int i = 0; i < 4; ++i)
	{
		cudaStreamDestroy(stream[i]);
	}
	cudaFree(boltz_D);
	cudaFree(s1_D);
	cudaFree(s2_D);
	cudaFree(J_x_D);
	cudaFree(J_y_D);
	cudaFree(J_z_D);

	cudaFree(M_buf_D);
	cudaFree(EJ_buf_D);


	curandDestroyGenerator(gen2);
}

gpu_3d::~gpu_3d(){

}

void gpu_3d::sweep(){
	LastError();

	dim3 block(8,8,8);
 	dim3 grid(ceil(L/8.),ceil(L/8.),ceil(L/8.));

	metrpolis_3d<<<grid,block,0,stream[1]>>>(s1_D,s2_D,gen_d,boltz_D,L,0);

	metrpolis_3d<<<grid,block,0,stream[1]>>>(s2_D,s1_D,gen_d,boltz_D,L,0);

	LastError();
}


void gpu_3d::set_T(double T_){
	T=T_;
	float boltz_H[14]; //boltzmanfaktor Host

	#ifndef SPLIT_H
	for (size_t i = 0; i < 7; i++) {
    int j=(i-3)*2;
    boltz_H[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
    boltz_H[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
  }
	#else
	for (int i = 0; i < 3; ++i)
	{
		int j=2+2*i;
		boltz_H[i] = exp(-2.*j*(1./T));
	}
	boltz_H[3]=exp(-2*h/T);
	#endif
	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(boltz_D, boltz_H, 14*sizeof(float), cudaMemcpyHostToDevice);
	// cudaBindTexture(0, get_3d_boltz(), boltz_D, 4*sizeof(float));
}

void gpu_3d::set_h(double h_){
	h=h_;
	float boltz_H[14]; //boltzmanfaktor Host

	#ifndef SPLIT_H
	for (size_t i = 0; i < 7; i++) {
		int j=(i-3)*2;
		boltz[i]    = std::min(1., exp(-2*(1/T)*(j-h)));
		boltz[7+i]  = std::min(1., exp(-2*(1/T)*(j+h)));
	}
	#else
	for (int i = 0; i < 3; ++i)
	{
		int j=2+2*i;
		boltz_H[i] = exp(-2.*j*(1./T));
	}
	boltz_H[3]=exp(-2*h/T);
	#endif

	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(boltz_D, boltz_H, 14*sizeof(float), cudaMemcpyHostToDevice);

	LastError();

}

vector<float> gpu_3d::measure(){
	int c=(int)ceil(N/(256.*2));
	float *M_H=new float[2*64*c];
	float *EJ_H=new float[2*64*c];
	LastError();
	dim3 grid(ceil(N/(2*256.)),1,1);
	dim3 block(256,1,1);
	dim3 block_s(8,8,8);
 	dim3 grid_s(ceil(L/8.),ceil(L/8.),ceil(L/8.));
	checkerbord_switch_3d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L,L*L);
	cudaStreamSynchronize(stream[1]);
	measure_EJ_M_3d<<<grid,block,0,stream[1]>>>(s1_D,&EJ_buf_D[0],&M_buf_D[0],N,L);
	measure_EJ_M_3d<<<grid,block,0,stream[2]>>>(s2_D,&EJ_buf_D[64*c],&M_buf_D[64*c],N,L);
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	checkerbord_switch_3d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L,L*L);
	cudaMemcpyAsync(&M_H[0],&M_buf_D[0],2*c*64*sizeof(float),cudaMemcpyDeviceToHost,stream[2]);
	cudaMemcpyAsync(&EJ_H[0],&EJ_buf_D[0],2*c*64*sizeof(float),cudaMemcpyDeviceToHost,stream[2]);
	LastError();
	cudaStreamSynchronize(stream[2]);

	vector<float> result;
	result.assign(2*64, 0);
	for (int i = 0; i < 2*64; ++i)
	{
		result[i]=0;
		for (int j = 0; j < c;++j)
		{
			result[i]+=EJ_H[i*c+j]/*/N*/+h*M_H[i*c+j]/*/N*/;
		}
	}
	delete[] M_H;
	delete[] EJ_H;
	return result;

}

void gpu_3d::set_seed(long long seed_){
	LastError();
	setup_radome<<<ceil(N/256.),256>>>(gen_d,seed_,N);
	curandSetPseudoRandomGeneratorSeed(gen2, seed_);

}

void gpu_3d::load_J(string fname){
	spin_t *Jx = new spin_t[2*N];
	spin_t *Jy = new spin_t[2*N];
	spin_t *Jz = new spin_t[2*N];
	int error=load_J_3d(Jx,Jy,Jz,N,fname);
	if(error<0){
		LOG(LOG_WARNING)<<"fehler beim laden von J in file \""<<fname<<"\" mit fehler "<<error<<endl;
		return;
	}
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(J_x_D, Jx, 2*N*sizeof(spin_t), cudaMemcpyHostToDevice);
	cudaMemcpy(J_y_D, Jy, 2*N*sizeof(spin_t), cudaMemcpyHostToDevice);
	cudaMemcpy(J_z_D, Jz, 2*N*sizeof(spin_t), cudaMemcpyHostToDevice);
	cudaBindTexture(0, get_3d_J_xi(), J_x_D,2*N*sizeof(int2));
	cudaBindTexture(0, get_3d_J_yi(), J_y_D,2*N*sizeof(int2));
	cudaBindTexture(0, get_3d_J_zi(), J_z_D,2*N*sizeof(int2));
	LastError();
	delete[] Jx;
	delete[] Jy;
	delete[] Jz;
}

void gpu_3d::save_J(string fname){
	spin_t *Jx = new spin_t[2*N];
	spin_t *Jy = new spin_t[2*N];
	spin_t *Jz = new spin_t[2*N];
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(Jx, J_x_D, 2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
	cudaMemcpy(Jy, J_y_D, 2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
	cudaMemcpy(Jz, J_z_D, 2*N*sizeof(spin_t),cudaMemcpyDeviceToHost);
    int error=save_J_3d(Jx,Jy,Jz,N,fname);
	if(error<0){
		LOG(LOG_WARNING)<<"fehler beim laden von J in file \""<<fname<<"\" mit fehler "<<error<<endl;
	}
	delete[] Jx;
	delete[] Jy;
	delete[] Jz;
}

void gpu_3d::init_J(){
	LastError();
	unsigned int *buffer_D; //Buffer divice
	gpuErrchk(cudaMalloc(&buffer_D, 3*2*N*sizeof(unsigned int)));
	//ranomly initlize J
	gpuErrchk(cudaDeviceSynchronize());
	generate_kernel<<<ceil(N/256.),256>>>(gen_d,buffer_D,N, 3*2);
	gpuErrchk(cudaDeviceSynchronize());
	J_order_3d<<<ceil(N/256.),256>>>(J_x_D, J_y_D, J_z_D, buffer_D, L, N);
	LastError();
	gpuErrchk(cudaDeviceSynchronize());
	cudaFree(buffer_D);
	LastError();
	cudaBindTexture(0, get_3d_J_xi(), J_x_D,2*N*sizeof(int2));
	LastError();
	cudaBindTexture(0, get_3d_J_yi(), J_y_D,2*N*sizeof(int2));
	LastError();
	cudaBindTexture(0, get_3d_J_zi(), J_z_D,2*N*sizeof(int2));
	LastError();

}

void gpu_3d::init_rand(){
	curandGenerateLongLong(gen2,s1_D,N);
	curandGenerateLongLong(gen2,s2_D,N);
	gpuErrchk(cudaDeviceSynchronize());
	LastError();

}

void gpu_3d::save_sys(string prefix){
	spin_t s[N];
	gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(s,s1_D,N*sizeof(spin_t),cudaMemcpyDeviceToHost);

	for (int i = 0; i < 64; ++i)
	{
		// image setup
		stringstream convert;
		convert<<prefix<<i<<".pbm";
		ofstream file(convert.str().c_str());
		file<<"P1"<<endl;
		file<<L<<" "<<L*L<<endl;
		// print image
		for (int j = 0; j < L*L; ++j)
		{
			for (int k = 0; k < L; ++k)
			{
				file<<((s[j*L+k]&((spin_t)1<<i))==0?"0 ":"1 ");
			}
			file<<endl;
		}
	}
}

long gpu_3d::get_N(){
	return N;
}

void gpu_3d::swap(gpu_sys *sys, std::unique_ptr<spin_t[]> mask) {
  gpu_3d *sys_3d = dynamic_cast<gpu_3d *>(sys);
  if (sys_3d != NULL) {
    swap(sys_3d, move(mask));
  } else {
    LOG(LOG_ERROR) << "conversion error";
  }
}

void gpu_3d::swap(gpu_3d *sys, std::unique_ptr<spin_t[]> mask) {
	dim3 block_s(8,8,8);
 	dim3 grid_s(ceil(L/8.),ceil(L/8.),ceil(L/8.));
	checkerbord_switch_3d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L,L*L);
	checkerbord_switch_3d<<<grid_s,block_s,0,stream[1]>>>(sys->s1_D,sys->s2_D,sys->L,sys->L*sys->L);
	swap_3d<<<ceil(N/256.),256,0,stream[1]>>>(s1_D,sys->s1_D,mask[0],N);
	swap_3d<<<ceil(N/256.),256,0,stream[1]>>>(s2_D,sys->s2_D,mask[1],N);
	checkerbord_switch_3d<<<grid_s,block_s,0,stream[1]>>>(sys->s1_D,sys->s2_D,sys->L,sys->L*sys->L);
	checkerbord_switch_3d<<<grid_s,block_s,0,stream[1]>>>(s1_D,s2_D,L,L*L);
}

ostream & gpu_3d::save(ostream &stream){
	binary_write(stream, L, 1);
	binary_write(stream, T, 1);
	binary_write(stream, h, 1);
	spin_t *s=new spin_t[N];
	cudaMemcpy(s,s1_D,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
	binary_write(stream, s[0], N);
	cudaMemcpy(s,s2_D,N*sizeof(spin_t),cudaMemcpyDeviceToHost);
	binary_write(stream, s[0], N);
	delete[] s;
	spin_t *J=new spin_t[2*N];
	cudaMemcpy(J,J_x_D,2*N*sizeof(int2),cudaMemcpyDeviceToHost);
	binary_write(stream, J[0], 2 * N);//x
	cudaMemcpy(J,J_y_D,2*N*sizeof(int2),cudaMemcpyDeviceToHost);
	binary_write(stream, J[0], 2 * N);//y
	cudaMemcpy(J,J_z_D,2*N*sizeof(int2),cudaMemcpyDeviceToHost);
	binary_write(stream, J[0], 2 * N);//z
	delete[] J;
	// RNG state
	curandStatePhilox4_32_10_t *gen_n_save=new curandStatePhilox4_32_10_t[N];
	cudaMemcpy(gen_n_save,gen_d,N*sizeof(curandStatePhilox4_32_10_t),cudaMemcpyDeviceToHost);
	binary_write(stream, gen_n_save[0], N);
	delete[] gen_n_save;

	return stream;
}

istream & gpu_3d::load(istream &stream){
	binary_read(stream, L, 1);
  N = L * L * L;
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
	cudaFree(J_z_D);
	cudaFree(M_buf_D);
	cudaFree(EJ_buf_D);
	cudaMalloc(&gen_d,N*sizeof(curandStatePhilox4_32_10_t));
	cudaMalloc(&s1_D, N*sizeof(spin_t));
	cudaMalloc(&s2_D, N*sizeof(spin_t));
	cudaMalloc(&J_x_D, 2*N*sizeof(int2));
	cudaMalloc(&J_y_D, 2*N*sizeof(int2));
	cudaMalloc(&J_z_D, 2*N*sizeof(int2));
	cudaMalloc(&M_buf_D, 2*64*ceil(N/256.+1)*sizeof(float));
	cudaMalloc(&EJ_buf_D, 2*64*ceil(N/256.+1)*sizeof(float));
	// spins
	spin_t *s=new spin_t[N];
	binary_read(stream, s[0], N);
	cudaMemcpy(s1_D,s,N*sizeof(spin_t),cudaMemcpyHostToDevice);
	binary_read(stream, s[0], N);
	cudaMemcpy(s2_D,s,N*sizeof(spin_t),cudaMemcpyHostToDevice);
	delete[] s;
	LastError();
	// copling
	spin_t *J=new spin_t[2*N];
	binary_read(stream, J[0], 2 * N);//x
	cudaMemcpy(J_x_D,J,2*N*sizeof(int2),cudaMemcpyHostToDevice);
	binary_read(stream, J[0], 2 * N);//y
	cudaMemcpy(J_y_D,J,2*N*sizeof(int2),cudaMemcpyHostToDevice);
	binary_read(stream, J[0], 2 * N);//y
	cudaMemcpy(J_z_D,J,2*N*sizeof(int2),cudaMemcpyHostToDevice);
	delete[] J;
	LastError();
	cudaBindTexture(0, get_3d_J_xi(), J_x_D,2*N*sizeof(int2));
	LastError();
	cudaBindTexture(0, get_3d_J_yi(), J_y_D,2*N*sizeof(int2));
	LastError();
	cudaBindTexture(0, get_3d_J_zi(), J_z_D,2*N*sizeof(int2));
	LastError();
	// RNG state
	curandStatePhilox4_32_10_t *gen_n_save=new curandStatePhilox4_32_10_t[N];
	binary_read(stream, gen_n_save[0], N);
	cudaMemcpy(gen_d,gen_n_save,N*sizeof(curandStatePhilox4_32_10_t),cudaMemcpyHostToDevice);
	delete[] gen_n_save;
	LastError();
	return stream;
}
