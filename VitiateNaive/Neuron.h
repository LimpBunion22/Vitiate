#pragma once
#include "Parameters.h"
#include <vector>
#include <fstream>

class ofstream;

class Neuron
{
private:
	uchar algoIndex; //índice de selección de algoritmo
	uint inputNum;	//número de entradas
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

	N_TYPE Algoritmo(const std::vector<N_TYPE>& inputs); //algoritmo empleado en la función de activación
	void ChangeAlgo(uint algoIndex); //selección del algoritmo a utilizar en la función de activación
	float Alfa(N_TYPE e); //derivada del algoritmo de la función de activación
	uint GetInputNum(); //getter
	const std::vector<N_TYPE>& GetCoefs(); //getter
	void PrintCoefs();
	void WriteCoefs(std::ofstream &coefsFile, uint layer, uint neuronPos);
};
