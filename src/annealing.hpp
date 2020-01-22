#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include "Logging.hpp"
#include "tools_inline.hpp"

template <class system>
class annealing
{
private:
	double T;

	double alfa;
	int steps;
	std::ostream *file;
	system *sys;
	std::vector<float> min_E;
	std::vector<float> result;
public:
	annealing(double T_s,int steps_,double alfa_, system *S,std::ostream &file_=std::cout);
	~annealing();
	void step(int stride, print_select select=PRINT_E);
	void print(print_select select);
	double get_T();
	void set_T(double T_s);
};


/**
 * @brief constructor
 * @details constructor of the annealing class
 *
 * @param T_s starting temperatur
 * @param steps_ number of sweeps betwin cooling
 * @param alfa_ cooling ratio
 * @param S system
 * @param file_ pointer to filestream defalet cout
 */
template <class system>
annealing<system>::annealing(double T_s,int steps_,double alfa_, system *S,std::ostream &file_){
	T=T_s;
	steps=steps_;
	sys=S;
	alfa=alfa_;
	file=&file_;
	sys->set_T(T);
	if(alfa>1&&alfa<0){
		throw("alfa_out_of_range");
	}
}

template <class system>
annealing<system>::~annealing(){

}

/**
 * @brief annealing step
 * @details simulast the system for one annealing step
 *
 * @param stride distinz betwen measuremens
 * @param select selects which to print out
 */
template <class system>
void annealing<system>::step(int stride, print_select select){
	for (int i = 0; i < steps; ++i)
	{
		sys->sweep();
		if (stride!=0&&i%stride==0)
		{
			result =	sys->measure();
			LOG(LOG_DEBUG)<<result.size();
			if(min_E.empty())
			{
				min_E=result;
			}
			if(min_E.size()==result.size())
			{
				for (unsigned int j = 0; j < min_E.size(); ++j)
				{
					min_E[j]=std::max(result[j], min_E[j]);
				}
			}
			else
			{
				LOG(LOG_ERROR)<<"min_E and result are not the same size";
			}
			if(select==PRINT_E){
				print(select);
			}
		}
	}
	T=T*alfa;

	sys->set_T(T);

}



template <class system>
void annealing<system>::print(print_select select){
	std::vector<float> data;
	switch (select){
		case PRINT_E: data=result; break;
		case PRINT_MIN: data=min_E; break;
		default: data=result; LOG(LOG_ERROR)<<"print select error";
	}
	LOG(LOG_DEBUG)<<data.size()<<"\t"<<result.size();
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		(*file)<<data[i]<<"\t";
	}
	(*file)<<std::endl;
}

/**
 * @brief get temperatur
 * @details gets temperatur of system
 * @return temperatur
 */
template <class system>
double annealing<system>::get_T(){
	return T;
}

/**
 * @brief seting T and resets min_E
 *
 * @param T_s Temperatur
 */
template <class system>
void annealing<system>::set_T(double T_s){
	T=T_s;
	min_E.clear();
}
