#pragma once
#include "Parameters.h"
#include "Neuron.h"
#include <vector>


class RED
{
private:
	uint layerNum;
	std::vector<std::vector<Neuron>> layers;

public:
	RED() = delete;
	RED(uint inputNum, uint* neuronsPerLayer, uint layerNum);
	RED(const RED& rhs);
	RED(RED&& rhs) noexcept;
	RED& operator = (const RED& rhs);
	RED& operator = (RED&& rhs) noexcept;

	std::vector<N_TYPE> Forward(std::vector<N_TYPE> inputs);
	void GetCoefs();
};

