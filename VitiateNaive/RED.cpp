#include "RED.h"
#include <time.h>
#include <iostream>

RED::RED(uint inputNum, uint* neuronsPerLayer, uint layerNum) : layerNum(layerNum), inputNum(inputNum), layers(layerNum)
{
	srand(time(NULL));

	for (uint i = 0;i < layerNum;i++)
	{
		this->neuronsPerLayer[i] = neuronsPerLayer[i];
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
		inputNum = rhs.inputNum;
		for (uint i = 0; i < layerNum; i++)
			neuronsPerLayer[i] = rhs.neuronsPerLayer[i];
		layers = rhs.layers;
	}

	return *this;
}

RED& RED::operator=(RED&& rhs) noexcept
{
	if (this != &rhs)
	{
		layerNum = rhs.layerNum;
		inputNum = rhs.inputNum;
		for (uint i = 0; i < layerNum; i++)
			neuronsPerLayer[i] = rhs.neuronsPerLayer[i];
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

		/*std::cout << "salida intermedia " << i << std::endl;

		for (auto j : tmpOuts)
			std::cout << j << " ";

		std::cout << std::endl;*/
	}

	return tmpOuts;
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


