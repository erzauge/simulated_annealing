#include <fstream>
#include <string>
#include <iostream>

#include "sys_file.hpp"
#include "bin_io.hpp"

using namespace std;

int load_J_2d( unsigned long long *Jx,unsigned long long *Jy,long N, string fname)
{
	ifstream file(fname,ios::in|ios::binary);

	if(!file.is_open())
	{
		return -1;
	}

	int d;
	binary_read(file,d,1);
	if (d!=2)
	{	
		return -2;
	}

	long n;
	binary_read(file,n,1);
	if (n!=N)
	{
		return -3;
	}

	binary_read(file,Jx[0],2*N);
	binary_read(file,Jy[0],2*N);
	file.close();
	return 0;
}


int save_J_2d( unsigned long long *Jx, unsigned long long *Jy,long N, string fname)
{
	ofstream file(fname,ios::out|ios::binary);

	if(!file.is_open())
	{
		return -1;
	}

	int d=2;
	binary_write(file,d,1);

	binary_write(file,N,1);

	binary_write(file,Jx[0],2*N);
	binary_write(file,Jy[0],2*N);
	file.close();
	return 0;
}


int load_J_3d(unsigned long long *Jx,unsigned long long *Jy,unsigned long long *Jz,long N, string fname)
{
	ifstream file(fname,ios::in|ios::binary);

	if(!file.is_open())
	{
		return -1;
	}

	int d;
	binary_read(file,d,1);
	if (d!=3)
	{	
		return -2;
	}

	long n;
	binary_read(file,n,1);
	if (n!=N)
	{
		return -3;
	}

	binary_read(file,Jx[0],2*N);
	binary_read(file,Jy[0],2*N);
	binary_read(file,Jz[0],2*N);
	file.close();
	return 0;
}

int save_J_3d(unsigned long long *Jx,unsigned long long *Jy,unsigned long long *Jz,long N, string fname)
{
	ofstream file(fname,ios::out|ios::binary);

	if(!file.is_open())
	{
		return -1;
	}

	int d=3;
	binary_write(file,d,1);

	binary_write(file,N,1);

	binary_write(file,Jx[0],2*N);
	binary_write(file,Jy[0],2*N);
	binary_write(file,Jz[0],2*N);
	file.close();
	return 0;
}
