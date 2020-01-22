#pragma once
typedef unsigned long long spin_t;
// inspierd by http://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
inline void swap_bits(spin_t &a, spin_t &b,spin_t &mask){
	spin_t r_a = a ^ ((a ^ b) & mask);
	spin_t r_b = b ^ ((b ^ a) & mask);
	a=r_a;
	b=r_b;
}

enum print_select
{
	PRINT_E,
	PRINT_MIN
};

//stupid idear but it works samehow
/**
 * @brief bit spin to double
 * @details locks if bit is set and returns depending on it -1 or 1
 *
 * @param s spin
 * @param j selected bit
 *
 * @return -1.0 if bit j in s is set else 1.0
 */
inline double bit_to_double(spin_t s, int j){
	spin_t a=(s<<j&((spin_t)1UL<<63))|0x3FF0000000000000; //0x3FF0000000000000 in double = 1. If bit j in s is set sets sing bit in double
	void * b= &a; //casting to void pinter
	return *((double*) b); //
}
