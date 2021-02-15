#pragma once
#include "Parameters.h"
#include "Neuron.h"
#include <vector>

class Matrix;

class RED
{
private:
	uint inputNum;
	uint layerNum;
	std::vector<std::vector<Neuron>> layers;

	void Forward(std::vector<N_TYPE> inputs, std::vector<std::vector<N_TYPE>>& e); //guarda cosas intermedias


public:
	RED() = delete;
	RED(uint inputNum, uint* neuronsPerLayer, uint layerNum);
	RED(const RED& rhs);
	RED(RED&& rhs) noexcept;
	RED& operator = (const RED& rhs);
	RED& operator = (RED&& rhs) noexcept;

	std::vector<N_TYPE> Forward(std::vector<N_TYPE> inputs); //ejecuta todo de forma continua sin guardar cosas intermedias
	void BuildMatrix(Matrix& A, Matrix& C, uint layer, std::vector<std::vector<N_TYPE>>&e);
	std::vector<std::vector<std::vector<N_TYPE>>> Gradient(std::vector<N_TYPE> inputs, std::vector<N_TYPE> s);
	void PrintCoefs();
	void PrintGradient(const std::vector<std::vector<std::vector<N_TYPE>>>& gradient);
};

