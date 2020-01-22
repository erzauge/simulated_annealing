#pragma once

template <class T>
class accumulator
{
private:
	T data;
	long long N;
public:
	accumulator();
	void operator()(T x, long long n=1);
	double get();
	void reset();
};

template <class T>
accumulator<T>::accumulator(){
	data=0;
	N=0;
}

template <class T>
void accumulator<T>::operator()(T x, long long n){
	data+=x;
	N+=n;
}

template <class T>
double accumulator<T>::get(){
	if(data!=0){
		return (double)(data/(double)N);
	}
	return 0.;
}

template <class T>
void accumulator<T>::reset(){
	accumulator();
}