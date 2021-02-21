#pragma once
#include "Parameters.h"
#include <vector>
#include <fstream>

class ofstream;

class Neuron
{
private:
	uchar algoIndex; //�ndice de selecci�n de algoritmo
	uint inputNum;	//n�mero de entradas
	N_TYPE sumatorio = 0; //var aux
	N_TYPE beta = (N_TYPE)((float)rand() / (RAND_MAX + 1) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE); //coef independiente
	std::vector<N_TYPE> coefs; //vector de coeficientes

public:
	Neuron() = delete;
	Neuron(uint inputNum, uint algoIndex);
	Neuron(const Neuron& rhs);
	Neuron(Neuron&& rhs) noexcept;
	Neuron& operator =(const Neuron& rhs);
	Neuron& operator =(Neuron&& rhs) noexcept;

	N_TYPE Algoritmo(const std::vector<N_TYPE>& inputs); //algoritmo empleado en la funci�n de activaci�n
	void ChangeAlgo(uint algoIndex); //selecci�n del algoritmo a utilizar en la funci�n de activaci�n
	float Alfa(N_TYPE e); //derivada del algoritmo de la funci�n de activaci�n
	uint GetInputNum(); //getter
	const std::vector<N_TYPE>& GetCoefs(); //getter
	void PrintCoefs();
	void WriteCoefs(std::ofstream &coefsFile, uint layer, uint neuronPos);
};
