#pragma once
#include "Parameters.h"
#include "Neuron.h"
#include <vector>
#include <windows.h>

class Matrix;

class RED
{
private:
	std::vector<N_TYPE>inputs; //*entradas de la red
	std::vector<N_TYPE>s; //*s: valores de aprendizaje
	uint layerNum; //número de capas totales de la red (entrada, salida, internas)
	std::vector<std::vector<Neuron>> layers; //vector (capa) de vector de neuronas
	LARGE_INTEGER microsFwd,microsGradient,freq; //control de tiempos de ejecución
	void Forward(std::vector<N_TYPE> inputs, std::vector<std::vector<N_TYPE>>& e); //Forward interno que guarda valores intermedios para su posterior uso en el backwards
	void BuildMatrix(Matrix& A, Matrix& C, uint layer, std::vector<std::vector<N_TYPE>>&e); //*genera la matriz a partir de 
	//*A: matriz diagonal cuyos elementos son las derivadas parciales de las funciones de activación
	//*C: matriz de coeficientes
	//*layer: número de capa
	//*e: valores intermedios calculados en el Forward

public:
	RED() = delete;
	RED(std::vector<N_TYPE> &inputs, std::vector<N_TYPE> &s, std::vector<N_TYPE> &neuronsPerLayer); //número de entradas, vector con número de neuronas por capa, número de capas
	RED(const RED& rhs);
	RED(RED&& rhs) noexcept;
	RED& operator = (const RED& rhs);
	RED& operator = (RED&& rhs) noexcept;

	std::vector<N_TYPE> Forward(std::vector<N_TYPE> inputs); //Forward que ejecuta todo de forma continua sin guardar valores
	std::vector<std::vector<std::vector<N_TYPE>>> Gradient(); 
	void PrintCoefs(); //getters
	void PrintGradient(const std::vector<std::vector<std::vector<N_TYPE>>>& gradient);
};

