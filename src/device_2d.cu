#include "device_2d.cuh"
#include "device.cuh"

#include <curand_kernel.h>
#include <curand.h>


#include <stdio.h>


texture<int2, cudaTextureType1D, cudaReadModeElementType> J_xi;

texture<int2, cudaTextureType1D, cudaReadModeElementType> J_yi;



texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_J_xi(void){
	return J_xi;
}

texture<int2, cudaTextureType1D, cudaReadModeElementType> &get_J_yi(void){
	return J_yi;
}

__forceinline__ __device__ __host__ int xy_id(int x, int y, int L){return y*L+x;}

__forceinline__ __device__  void updat_spin_split(spin_t &s_i, spin_t &s_w, spin_t &s_n, spin_t &s_e, spin_t &s_s, spin_t border, float &rand1, float &rand2, spin_t &Jw, spin_t &Jn, spin_t &Je, spin_t &Js,float * boltz)
{
 	spin_t sw,sn,se,ss,p0,p1,mask,d0,d1,d01,d11,d00,d10,dh,mh;
 	sw =Jw^s_i^s_w;
	sn =Jn^s_i^s_n;
	se =Je^s_i^s_e;
	ss =Js^s_i^s_s;

	p0= sw&sn&se&ss;
	p1=((sw^sn)&se&ss)|(sw&sn&(se^ss));

	mask=~(spin_t)0;
	d01	= rand1<boltz[3-1]?mask:(spin_t)0;//nicht perodischer rand
	d11	= rand1<boltz[1-1]?mask:(spin_t)0;//nicht perodischer rand
	d00	= rand1<boltz[4-1]?mask:(spin_t)0;
	d10	= rand1<boltz[2-1]?mask:(spin_t)0;
	dh	= rand2<boltz[4]?mask:(spin_t)0;

	d0 	=(d00|border)&(d01|~border);
	d1 	=(d10|border)&(d11|~border);
	mh 	= ~s_i|dh;

	s_i^= ((p0&d0)|(p1&d1)|~(p0|p1))&mh;
}


__forceinline__ __device__  void updat_spin(spin_t &s_i, spin_t &s_w, spin_t &s_n, spin_t &s_e, spin_t &s_s, float &rand1, spin_t &Jw, spin_t &Jn, spin_t &Je, spin_t &Js,float * boltz)
{

 	spin_t sw =Jw^s_i^s_w;
	spin_t sn =Jn^s_i^s_n;
	spin_t se =Je^s_i^s_e;
	spin_t ss =Js^s_i^s_s;

	spin_t p0 = ((~sw)&(~se)&(~sn)&~ss);
  spin_t p1 = ((sw^sn)&~se&~ss)|(~sw&~sn&(se^ss));
  spin_t p2 = ((sw ^ sn)&(se ^ ss))|((sw ^ ss)&(se ^ sn));
  spin_t p3 = ((sw^sn)&se&ss)|(sw&sn&(se^ss));
  spin_t p4 = sw&se&sn&ss;

	spin_t mask=~(spin_t)0;
	spin_t d00 = rand1 < boltz[0]? mask : (spin_t)0;
	spin_t d10 = rand1 < boltz[1]? mask : (spin_t)0;
	spin_t d20 = rand1 < boltz[2]? mask : (spin_t)0;
	spin_t d30 = rand1 < boltz[3]? mask : (spin_t)0;
	spin_t d40 = rand1 < boltz[4]? mask : (spin_t)0;
	spin_t d01 = rand1 < boltz[0+5]? mask : (spin_t)0;
	spin_t d11 = rand1 < boltz[1+5]? mask : (spin_t)0;
	spin_t d21 = rand1 < boltz[2+5]? mask : (spin_t)0;
	spin_t d31 = rand1 < boltz[3+5]? mask : (spin_t)0;
	spin_t d41 = rand1 < boltz[4+5]? mask : (spin_t)0;

	spin_t h 	= s_i;
	spin_t d0 = (d00 | h) & (d01 | ~h);
	spin_t d1 = (d10 | h) & (d11 | ~h);
	spin_t d2 = (d20 | h) & (d21 | ~h);
	spin_t d3 = (d30 | h) & (d31 | ~h);
	spin_t d4 = (d40 | h) & (d41 | ~h);

	s_i^= (d0&p0)|(d1&p1)|(d2&p2)|(d3&p3)|(d4&p4);;
}


__global__ void metrpolis_2d(spin_t *s_u,spin_t *s_i,curandStatePhilox4_32_10_t *random,float * boltz,int L, long J_offset){
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=threadIdx.y+blockIdx.y*blockDim.y;

	if(x<L&&y<L)
	{
		curandStatePhilox4_32_10_t rng=random[xy_id(x,y,L)];
		float rand1 = curand_uniform(&rng);
		#ifdef SPLIT_H
		float rand2 = curand_uniform(&rng);
		#endif
		random[xy_id(x,y,L)]=rng;
		spin_t Jw = int2_as_longlong(tex1Dfetch(J_xi,J_offset+2*xy_id(x,y,L)));
		spin_t Jn = int2_as_longlong(tex1Dfetch(J_yi,J_offset+2*xy_id(x,y,L)));
		spin_t Je = int2_as_longlong(tex1Dfetch(J_xi,J_offset+2*xy_id(x,y,L)+1));
		spin_t Js = int2_as_longlong(tex1Dfetch(J_yi,J_offset+2*xy_id(x,y,L)+1));
		if (x!=0&&y!=0&&x!=(L-1)&&y!=(L-1))
		{
			#ifdef SPLIT_H
			updat_spin_split(s_u[xy_id(x,y,L)],s_i[xy_id(x-1,y,L)],s_i[xy_id(x,y-1,L)],s_i[xy_id(x+1,y,L)],s_i[xy_id(x,y+1,L)],(spin_t)0,rand1,rand2, Jw, Jn, Je, Js,boltz);
			#else
			updat_spin(s_u[xy_id(x,y,L)],s_i[xy_id(x-1,y,L)],s_i[xy_id(x,y-1,L)],s_i[xy_id(x+1,y,L)],s_i[xy_id(x,y+1,L)],rand1, Jw, Jn, Je, Js,&boltz[0]);
			#endif
		}
		else
		{
			spin_t s_w = (x!=0)?s_i[xy_id(x-1,y,L)]:~((s_u[xy_id(x,y,L)]^Jw));
			spin_t s_n = (y!=0)?s_i[xy_id(x,y-1,L)]:s_i[xy_id(x,L-1,L)];
			spin_t s_e = (x!=(L-1))?s_i[xy_id(x+1,y,L)]:~((s_u[xy_id(x,y,L)]^Je));
			spin_t s_s = (y!=(L-1))?s_i[xy_id(x,y+1,L)]:s_i[xy_id(x,0,L)];
			#ifdef SPLIT_H
			spin_t border = (x!=0&&x!=(L-1))?(spin_t)0:~(spin_t)0;
			updat_spin_split(s_u[xy_id(x,y,L)],s_w,s_n,s_e,s_s,border,rand1,rand2, Jw, Jn, Je, Js,boltz);
			#else
			updat_spin(s_u[xy_id(x,y,L)],s_w,s_n,s_e,s_s,rand1, Jw, Jn, Je, Js,&boltz[10]);
			#endif
		}
	}
}

__global__ void J_order(int2 *J_xi_d, int2 *J_yi_d,unsigned int *buffer, int L, long N){
	long x=blockIdx.x*blockDim.x+threadIdx.x;
	if(x<N){
	J_xi_d[2*x]=make_int2(buffer[2*(2*x)],buffer[2*(2*x)+1]);
	J_xi_d[2*x+1]=(x%L==(L-1))?make_int2(buffer[2*(2*(x-L+1))],buffer[2*(2*(x-L+1))+1]):make_int2(buffer[2*(2*(x+1))],buffer[2*(2*(x+1))+1]);
	J_yi_d[2*x]=make_int2(buffer[2*(2*x+1)],buffer[2*(2*x+1)+1]);
	J_yi_d[2*x+1]=((x+L)>=N)?make_int2(buffer[2*(2*(x-N+L)+1)],buffer[2*(2*(x-N+L)+1)+1]):make_int2(buffer[2*(2*(x+L)+1)],buffer[2*(2*(x+L)+1)+1]);
	}
}


__global__ void measure_EJ_M_2d(spin_t *s_1, float * EJ_buf, float * M_buf, long N, int L, long J_start){

	__shared__ spin_t si[2*256];
	__shared__ spin_t Ew[2*256];
	__shared__ spin_t En[2*256];
	long id1 = 2*(blockIdx.x*blockDim.x+threadIdx.x);
	long id2 = 2*(blockIdx.x*blockDim.x+threadIdx.x)+1;

	int w_id=2*(threadIdx.x%32);

	int w_s=32*2*(threadIdx.x/32);

	si[w_s+w_id]=0;
	si[w_s+w_id+1]=0;
	Ew[w_s+w_id]=0;
	Ew[w_s+w_id+1]=0;
	En[w_s+w_id]=0;
	En[w_s+w_id+1]=0;
	if (id1<N)
	{
		spin_t si1 = s_1[id1];
		spin_t sw1 =((id1%L)==0)?~(spin_t)0:(int2_as_longlong(tex1Dfetch(J_xi,2*id1+J_start))^si1^s_1[id1-1]);
		spin_t sn1 = int2_as_longlong(tex1Dfetch(J_yi,2*id1+J_start))^si1^s_1[(id1<L)?id1+N-L:id1-L];

		spin_t si2 = s_1[id2];
		spin_t sw2 =((id2%L)==0)?~(spin_t)0:(int2_as_longlong(tex1Dfetch(J_xi,2*id2+J_start))^si2^s_1[id2-1]);
		spin_t sn2 = int2_as_longlong(tex1Dfetch(J_yi,2*id2+J_start))^si2^s_1[(id2<L)?id2+N-L:id2-L];
		int j1;

		//umorden der inhate inerhalb  eines warps/blocks.
		#pragma unroll
		for (int i = 0; i < 64; i++)
		{
			j1=(w_id+i)%64;

			si[w_s+j1]|=((si1>>j1)&((spin_t)1UL))<<w_id;
			Ew[w_s+j1]|=((sw1>>j1)&((spin_t)1UL))<<w_id;
			En[w_s+j1]|=((sn1>>j1)&((spin_t)1UL))<<w_id;

			si[w_s+j1]|=((si2>>j1)&((spin_t)1UL))<<w_id+1;
			Ew[w_s+j1]|=((sw2>>j1)&((spin_t)1UL))<<w_id+1;
			En[w_s+j1]|=((sn2>>j1)&((spin_t)1UL))<<w_id+1;
		}
	}
	__syncthreads();


	if (threadIdx.x<64)
	{
		int M=0,E=0;
		double over=0;
		#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			M+=__popcll(si[threadIdx.x+i*64]);
			E+=__popcll(Ew[threadIdx.x+i*64])+__popcll(En[threadIdx.x+i*64]);
		}

		if(((blockIdx.x+1)*512)>N)
		{
			over=(blockIdx.x+1)*512-N;
		}

		M_buf[threadIdx.x*gridDim.x+blockIdx.x]=(2.*M-8*64)+over;
		EJ_buf[threadIdx.x*gridDim.x+blockIdx.x]=(2.*E-16.*64.)+2*over;

	}
}

__global__ void checkerbord_switch_2d(spin_t *s_1,spin_t *s_2, int L){
	int x=threadIdx.x+blockIdx.x*blockDim.x;
	int y=2*threadIdx.y+blockIdx.y*blockDim.y+x%2;
	if(x<L&&y<L)
	{
		spin_t b=s_1[xy_id(x,y,L)];
		s_1[xy_id(x,y,L)]=s_2[xy_id(x,y,L)];
		s_2[xy_id(x,y,L)]=b;
	}

}

__global__ void swap_2d(spin_t *s_a,spin_t *s_b,spin_t mask,long N){
	long x=blockIdx.x*blockDim.x+threadIdx.x;
	if(x<N){
		swap_bits(s_a[x],s_b[x],mask);
	}
}
