#pragma once
#include <curand.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>

#include "setup.cuh"
#include "gpu_sys.cuh"
#include "random.cuh"

using namespace std;


class gpu_2d : public gpu_sys
{
private:
	spin_t *s1_D,*s2_D; //spins divice
	int2 *J_x_D, *J_y_D; //koplungs kostant divice


	curandGenerator_t /*gen,*/ gen2; //randome number generater

	curandStatePhilox4_32_10_t *gen_d;

	int num_u;
	int count_u;


	float *boltz_D; //boltzmanfaktor divice


	float *M_buf_D;
	float *EJ_buf_D;


	double T; // Tempertur
	double h; // magnetfeld

	int L; // system size
	long N; // number of spins

	cudaStream_t stream[4]; // cuda Streams





public:
	gpu_2d(int L_,double T_, double h_);
	void sweep() override;
	void set_T(double T_) override;
	void set_h(double h_) override;
	void set_seed(long long seed_) override;
	vector<float> measure() override;
	void gpu_free() override;
	void load_J(string fname) override;
	void save_J(string fname) override;
	void init_J() override;
	void init_rand() override;
	void save_sys(string prefix) override;
	long get_N() override;
	void swap(gpu_sys *sys, std::unique_ptr<spin_t[]> mask) override;
	void swap(gpu_2d *sys, std::unique_ptr<spin_t[]> mask);
	ostream & save(ostream &stream) override;
  istream & load(istream &stream) override;
	virtual ~gpu_2d();

};
