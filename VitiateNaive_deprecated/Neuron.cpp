#include "Neuron.h"
#include <math.h>
#include <iostream>

Neuron& Neuron::operator=(const Neuron& rhs)
{
#ifdef DEBUG
	std::cout << "copy operator" << std::endl;
#endif
	if (this != &rhs)
	{
		algoIndex = rhs.algoIndex;
		inputNum = rhs.inputNum;
		beta = rhs.beta;
		coefs = rhs.coefs;
	}

	return *this;
}

Neuron& Neuron::operator=(Neuron&& rhs) noexcept
{
#ifdef DEBUG
	std::cout << "move operator" << std::endl;
#endif
	if (this != &rhs)
	{
		algoIndex = rhs.algoIndex;
		inputNum = rhs.inputNum;
		beta = rhs.beta;
		coefs = rhs.coefs;
	}

	return *this;
}


Neuron::Neuron(uint inputNum, uint algoIndex) :algoIndex(algoIndex), inputNum(inputNum)
{
#ifdef DEBUG
	std::cout << "default ctr" << std::endl;
#endif
	coefs.reserve(inputNum); //se reserva memoria para que no tenga que realojarse cada vez que se añade un elemento

	for (uint i = 0; i < inputNum; i++)
		coefs.emplace_back((N_TYPE)((float)rand() / (RAND_MAX + 1) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE)); //coeficientes aleatorios
}

Neuron::Neuron(const Neuron& rhs)
{
#ifdef DEBUG
	std::cout << "copy ctr" << std::endl;
#endif
	* this = rhs;
}

Neuron::Neuron(Neuron&& rhs) noexcept
{
#ifdef DEBUG
	std::cout << "move ctr" << std::endl;
#endif
	* this = std::move(rhs);
}

N_TYPE Neuron::Algoritmo(const std::vector<N_TYPE>& inputs) //el primer input es 1, para multiplicar por beta
{
	sumatorio = 0;

	for (uint i = 0;i < inputNum;i++)
	{
		sumatorio += coefs[i] * inputs[i];
	}

	sumatorio += beta;

	switch (algoIndex)
	{
	case 0: //algo 2relu
		if (sumatorio < 0)
			sumatorio /= 256;
		break;
	case 1: //algo relu básica
		if (sumatorio < 0)
			sumatorio = 0;
		break;
		//case 2: //algo sigmoide
		//	sumatorio = (N_TYPE)(BASE / (1 + exp(-sumatorio)));
		//	break;
	default:
		std::cout << "algoritmo no definido" << std::endl;
		return 0;
		break;
	}

	return sumatorio;
}

void Neuron::ChangeAlgo(uint algoIndex)
{
	this->algoIndex = algoIndex;
}

float Neuron::Alfa(N_TYPE e)
{
	switch (algoIndex)
	{
	case 0: //algo 2relu
		return e < 0 ? (float)1 / 256 : 1;
		break;
	case 1: //algo relu básica
		return e < 0 ? 0 : 1;
		break;
		//case 2: //algo sigmoide
		//	return;
		//	break;
	default:
		std::cout << "algoritmo no definido" << std::endl;
		return 0;
		break;
	}

}

uint Neuron::GetInputNum()
{
	return inputNum;
}

const std::vector<N_TYPE>& Neuron::GetCoefs()
{
	return coefs;
}

void Neuron::PrintCoefs()
{
	std::cout << "coefs neurona" << std::endl;
	std::cout << beta << std::endl;

	for (auto i : coefs)
		std::cout << i << " ";

	std::cout << std::endl <<std::endl;
}
