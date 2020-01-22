#pragma once
#include <iostream>
#include <cmath>


class block
{
private:
	double start;
	double stop;
	double step;
public:
	block();
	block(double start_,double step_,double stop_);
	int size() const;
	double operator[](int i)const;
	void operator()(double *b);
	friend std::ostream& operator<<(std::ostream &,const block &);
	friend std::istream& operator>>(std::istream &,block &);
};
