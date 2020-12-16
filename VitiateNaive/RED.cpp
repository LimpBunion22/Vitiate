#include "RED.h"
#include <time.h>
#include <iostream>

RED::RED(uint inputNum, uint* neuronsPerLayer, uint layerNum) : inputNum(inputNum), layerNum(layerNum), layers(layerNum)
{
	srand(time(NULL));

	for (uint i = 0;i < layerNum;i++)
	{
		layers[i].reserve(neuronsPerLayer[i]); //neuronas por capa
		if (i)
			for (uint j = 0;j < neuronsPerLayer[i];j++)
				layers[i].emplace_back(neuronsPerLayer[i - 1], ACTIVATION_FUNC); // entradas por neurona, activación
		else
			for (uint j = 0;j < neuronsPerLayer[i];j++)
				layers[i].emplace_back(inputNum, ACTIVATION_FUNC);
	}
}

RED::RED(const RED& rhs)
{
	*this = rhs;
}

RED::RED(RED&& rhs) noexcept
{
	*this = std::move(rhs);
}

RED& RED::operator=(const RED& rhs)
{
	if (this != &rhs)
	{
		layerNum = rhs.layerNum;
		layers = rhs.layers;
	}

	return *this;
}

RED& RED::operator=(RED&& rhs) noexcept
{
	if (this != &rhs)
	{
		layerNum = rhs.layerNum;
		layers = rhs.layers;
	}

	return *this;
}

std::vector<N_TYPE> RED::Forward(std::vector<N_TYPE> inputs)
{
	std::vector<N_TYPE> tmpOuts;

	for (uint i = 0; i < layerNum; i++)
	{
		tmpOuts.clear();
		tmpOuts.reserve(layers[i].size());

		for (auto& j : layers[i])
			tmpOuts.emplace_back(j.Algoritmo(inputs));

		inputs = tmpOuts;

#ifdef DEBUG
		std::cout << "salida intermedia " << i << std::endl;

		for (auto j : tmpOuts)
			std::cout << j << " ";

		std::cout << std::endl;
#endif
	}

	return tmpOuts;
}

void RED::Forward(std::vector<N_TYPE> inputs, std::vector<std::vector<N_TYPE>>& e)
{
	std::vector<N_TYPE> tmpOuts;
	e.emplace_back(inputs); //entradas iniciales

	for (uint i = 0; i < layerNum; i++)
	{
		tmpOuts.clear();
		tmpOuts.reserve(layers[i].size());

		for (auto& j : layers[i])
			tmpOuts.emplace_back(j.Algoritmo(inputs));

		inputs = tmpOuts;
		e.emplace_back(inputs);
	}
}

std::vector<std::vector<std::vector<N_TYPE>>> RED::Gradient(std::vector<N_TYPE> inputs, std::vector<N_TYPE> s)
{
	std::vector<std::vector<std::vector<N_TYPE>>> output(layerNum);

	for (uint i = 0;i < layerNum;i++)
	{
		uint neuronNum = layers[i].size();
		output[i].reserve(neuronNum);

		for (uint j = 0;j < neuronNum;j++)
		{
			output[i].emplace_back();
			output[i][j].reserve((size_t)(layers[i][j].GetInputNum()) + 1);
		}
	}

	std::vector<std::vector<N_TYPE>> e((size_t)layerNum + 1);
	Forward(inputs, e);
	N_TYPE E = 0;

	for (uint i = 0; i < layers.back().size(); i++) //índice de neuronas de la última capa
	{
		E = 2 * (s[i] - e.back()[i]) * layers.back()[i].Alfa(e.back()[i]); //E*A
		output.back()[i][0] = E; //término independiente;

		for (uint j = 0; j < layers[(size_t)(layerNum - 2)].size(); j++) //-2=penúltima capa
		{
			output.back()[i][(size_t)j + 1] = E * e[(size_t)layerNum - 1][j]; //salidas de la última capa oculta. Como en e hay layerNum+1, la última capa es layerNum y la penúltima -1
			N_TYPE E2 = layers.back()[i].GetCoefs()[j] * layers[(size_t)(layerNum - 2)][j].Alfa(e[(size_t)(layerNum - 2)][j]); //wn*A de la última capa oculta
			output[(size_t)(layerNum - 2)][j][0] = E * E2; //término independiente;

			for (uint n = 0; n < layers[(size_t)(layerNum - 2)][0].GetInputNum(); n++) //cualquier neurona de la capa vale para el índice
			{
				output[(size_t)(layerNum - 2)][j][(size_t)n + 1] = E * E2 * e[(size_t)layerNum - 2][n]; //salidas de la penúltima capa
			}
		}
	} //capa de salidas y lloros

	return output;
}

void RED::GetCoefs()
{
	for (uint i = 0; i < layerNum;i++)
	{
		std::cout << "capa " << i << std::endl;

		for (auto& j : layers[i])
			j.GetCoefs();
	}

}


