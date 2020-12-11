#pragma once
#include "Parameters.h"
#include <vector>

class Neuron
{
private:
	uchar algoIndex;
	uint inputNum;
	N_TYPE sumatorio = 0;
	N_TYPE beta = (N_TYPE)((float)rand()/(RAND_MAX+1)*(MAX_RANGE-MIN_RANGE)+MIN_RANGE); //coef independiente
	std::vector<N_TYPE> coefs; 

public:
	Neuron() = delete;
	Neuron(uint inputNum, uint algoIndex);
	Neuron(const Neuron& rhs);
	Neuron(Neuron&& rhs) noexcept;
	Neuron& operator =(const Neuron& rhs);
	Neuron& operator =(Neuron&& rhs) noexcept;

	N_TYPE Algoritmo(const std::vector<N_TYPE>& inputs);
	void ChangeAlgo(uint algoIndex);
	void GetCoefs();
};
