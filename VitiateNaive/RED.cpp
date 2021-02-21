#include "RED.h"
#include <time.h>
#include <iostream>
#include "Matrix.h"
#include <fstream>

RED::RED(uint inputNum, uint* neuronsPerLayer, uint layerNum) : inputNum(inputNum), layerNum(layerNum), layers(layerNum)
{
	srand(time(NULL));

	std::ofstream coefsFile;
	coefsFile.open("coefsFile.txt");

	for (uint i = 0;i < layerNum;i++) //para cada capa
	{
		layers[i].reserve(neuronsPerLayer[i]); //reservar el número de neuronas por capa

		if (i) //si no es la primera capa
			for (uint j = 0;j < neuronsPerLayer[i];j++) //para cada neurona
			{
				layers[i].emplace_back(neuronsPerLayer[i - 1], ACTIVATION_FUNC); //crear neurona, siendo el número de entradas la cantidad de neuronas de la capa anterior
				layers[i].back().WriteCoefs(coefsFile,i,j);
			}
		else //si es la primera capa
			for (uint j = 0;j < neuronsPerLayer[i];j++) //para cada neurona 
			{
				layers[i].emplace_back(inputNum, ACTIVATION_FUNC); //crear neurona, siendo el número de entradas la cantidad de entradas de la red
				layers[i].back().WriteCoefs(coefsFile,i,j);
			}
	}

	coefsFile.close();
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

std::vector<N_TYPE> RED::Forward(std::vector<N_TYPE> inputs) //forward público
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

void RED::Forward(std::vector<N_TYPE> inputs, std::vector<std::vector<N_TYPE>>& e) //forward privado
{
	std::vector<N_TYPE> tmpOuts; //vector temporal en el que se guardan los resultados
	e.reserve((size_t)layerNum + 1); //reservamos número de capas + entradas
	e.emplace_back(inputs); //inputs iniciales

	for (uint i = 0; i < layerNum; i++)
	{
		tmpOuts.clear();
		tmpOuts.reserve(layers[i].size()); //número de neuronas por capa

		for (auto& j : layers[i]) //para cada neurona
			tmpOuts.emplace_back(j.Algoritmo(inputs)); //ejecutar algoritmo forward

		inputs = tmpOuts; //los cálculos obtenidos en la capa anterior son las entradas de la siguiente
		e.emplace_back(inputs); //se guardan los cálculos en el vector e
	}
}

void RED::BuildMatrix(Matrix& A, Matrix& C, uint layer, std::vector<std::vector<N_TYPE>>& e)
{
	std::vector<N_TYPE> zeroRow(layers[layer].size(), 0); //fila con todo ceros

	for (uint i = 0;i < layers[layer].size();i++)
	{
		C.PlaceRow(layers[layer][i].GetCoefs()); //matriz C de coeficientes 

		zeroRow[i] = layers[layer][i].Alfa(e[(size_t)layer + 1][i]); //layer+1 pq el tamaño de e es layerNum+1 (el primer vector es el de entradas de la red)
																	// Solo la posición "i" de zeroRow contendrá una derivada, el resto son ceros
																	//^Al ser primitivas, no debería volver a copiar todo si modificas algún valor
		A.PlaceRow(zeroRow); //matriz A diagonal de derivadas 
		zeroRow[i] = 0; //se resetea la fila con todo ceros
	}
}

std::vector<std::vector<std::vector<N_TYPE>>> RED::Gradient(std::vector<N_TYPE> inputs, std::vector<N_TYPE> s)
{
	std::vector<std::vector<std::vector<N_TYPE>>> output(layerNum);
	//*vector exterior: capas (externas e internas)
	//*vector intermedio: neuronas de la capa
	//*vector interno: valores de gradiente asociados a los coeficientes de cada neurona

	for (uint i = 0;i < layerNum;i++) //reserva y creación de los vectores internos de "output"
	{
		output[i].reserve(layers[i].size()); //para cada capa, se reserva el número de neuronas correspondiente

		for (uint j = 0; j < layers[i].size(); j++) //para cada neurona de la capa
			output[i].emplace_back((size_t)layers[i][j].GetInputNum() + 1, 0); 	//crear un vector igual al tamaño de coefs. + término independiente 
																				//inicializando valores del gradiente a 0 (valores de ese vector)
																				//^Al ser primitivas, no debería volver a copiar todo si modificas algún valor
	}

	std::vector<std::vector<N_TYPE>> e; //nº de capas + 1 para guardar los valores de inputs
	Forward(inputs, e);
	N_TYPE E = 0;
	std::vector<N_TYPE> Evec;
	std::vector<Matrix> outputRow(layers.back().size(), 1); //resevar 1 fila para cada matriz del vector de salidas

	for (uint i = 0; i < layers.back().size(); i++) //índice de neuronas de la última capa
	{
		E = -2 * (s[i] - e.back()[i]) * layers.back()[i].Alfa(e.back()[i]); //E*A
		//! introducimos un menos inicial resultante de derivar el error cuadrático. Cuando apliquemos los valores del gradiante, habrá que restarlos y no sumarlos
		output.back()[i][0] = E; //derivada del término independiente;

		for (uint j = 0; j < layers[(size_t)layerNum - 2].size(); j++) //-2=penúltima capa
		{
			output.back()[i][(size_t)j + 1] = E * e[(size_t)layerNum - 1][j]; //Derivada de cada coeficiente. Salidas de la última capa oculta (Como en e hay layerNum+1, la última capa es layerNum y la penúltima -1)
			N_TYPE E2 = layers.back()[i].GetCoefs()[j] * layers[(size_t)layerNum - 2][j].Alfa(e[(size_t)layerNum - 1][j]); //wn*A de la última capa oculta

			//? no debería hacer falta distinguir entre primera neurona y el resto, ya que todos los valores de output están inicializados a cero (no hay valores "extraños" que debamos sobreescribir)
			output[(size_t)layerNum - 2][j][0] += E * E2; //derivada del término independiente

			for (uint n = 0; n < layers[(size_t)layerNum - 2][0].GetInputNum(); n++) //cualquier neurona de la capa vale para el índice
				output[(size_t)layerNum - 2][j][(size_t)n + 1] += E * E2 * e[(size_t)layerNum - 2][n]; //Derivada de cada coeficiente. Salidas de la penúltima capa
		}

		//-------------------------------------------------------------
		//^ outputRow será utilizado en los subsiguientes cálculos de las demás capas
		Evec.reserve(layers.back()[0].GetInputNum()); //da igual qué neurona

		for (uint i2 = 0; i2 < layers.back()[0].GetInputNum(); i2++)
			Evec.emplace_back(layers.back()[i].GetCoefs()[i2] * E * layers[(size_t)layerNum - 2][i2].Alfa(e[(size_t)layerNum - 1][i2])); //! código modificado

		outputRow[i].PlaceRow(Evec);
		Evec.clear();
	} //capa de salidas y penúltima

	if (layerNum > 2) //si hay más de dos capas
	{
		for (uint i2 = 0; i2 < layers[(size_t)layerNum - 3].size();i2++)
		{
			float alfa = layers[(size_t)layerNum - 3][i2].Alfa(e[(size_t)layerNum - 2][i2]); //-3+1 pq tiene layerNum+1
			std::vector<N_TYPE> aux;
			aux.reserve(layers[(size_t)layerNum - 2].size());

			for (uint n = 0; n < layers[(size_t)layerNum - 2].size(); n++)
				aux.emplace_back(layers[(size_t)layerNum - 2][n].GetCoefs()[i2] * alfa); //para cada neurona (i2) de la capa i, guardas el coef i de cada neurona de la capa i+1

			Matrix col;
			col.ColVector(aux);

			for (uint i3 = 0; i3 < layers.back().size(); i3++) //nº de salidas de la red
			{
				N_TYPE val = (outputRow[i3] * col)[0][0]; //la multiplicación es una matriz, de la cual extraemos el primer término (fila 1, col 1)

			//? no debería hacer falta distinguir entre primera neurona y el resto, ya que todos los valores de output están inicializados a cero (no hay valores "extraños" que debamos sobreescribir)
				output[(size_t)layerNum - 3][i2][0] += val; //derivada del término independiente

				for (uint i4 = 0; i4 < layers[(size_t)layerNum - 3][i2].GetInputNum(); i4++) //+1 el término independiente
					output[(size_t)layerNum - 3][i2][(size_t)i4 + 1] += val * e[(size_t)layerNum - 3][i4];  //Derivada de cada coeficiente. +1 término independiente
					//! corregido error de derivada (f'(c)=e)
			}
		}
	}

	if (layerNum > 3) //si hay más capas que la de entrada, salida y 1 intermedia
	{
		Matrix result;

		for (int i = layerNum - 4; i >= 0; i--) //índice de neuronas de la última capa
		{	
			Matrix A(layers[i+2].size()); //! código modificado (matriz de la capa i+2)
			Matrix C(layers[i+2].size());
			BuildMatrix(A, C, i+2, e);

			if (i == layerNum - 4) //^ diferenciación ya que alfa ya se tiene en cuenta en output row, necesario para el cálculo if layerNum > 2. En las siguientes capas se procede normalmente
				result = C;
			else
				result = result * A * C;			

			for (uint i2 = 0; i2 < layers[i].size();i2++)
			{
				float alfa = layers[i][i2].Alfa(e[(size_t)i + 1][i2]); //i+1 pq tiene layerNum+1
				std::vector<N_TYPE> aux;
				aux.reserve(layers[(size_t)i + 1].size());

				for (uint n = 0; n < layers[(size_t)i + 1].size(); n++)
					aux.emplace_back(layers[(size_t)i + 1][n].GetCoefs()[i2] * alfa * layers[i+1][n].Alfa(e[(size_t)i + 2][n])); //para cada neurona (i2) de la capa i, guardas el coef i de cada neurona de la capa i+1
																																 //! se introduce alfa de la capa i+1
				Matrix col;
				col.ColVector(aux);
				Matrix result2 = result * col;

				for (uint i3 = 0; i3 < layers.back().size(); i3++) //nº de salidas de la red
				{
					N_TYPE val = (outputRow[i3] * result2)[0][0];

					output[i][i2][0] += val; //término independiente

					for (uint i4 = 0; i4 < layers[i][i2].GetInputNum(); i4++) //+1 el término independiente
					{
						output[i][i2][(size_t)i4 + 1] += val * e[(size_t)i][i4]; //+1 término independiente
					}				

				}
			}
		}
	}

	return output;
}

void RED::PrintCoefs()
{
	for (uint i = 0; i < layerNum;i++)
	{
		std::cout << "CAPA " << i << std::endl;

		for (auto& j : layers[i])
			j.PrintCoefs();
	}
}

void RED::PrintGradient(const std::vector<std::vector<std::vector<N_TYPE>>>& gradient)
{
	std::cout << "GRADIENTE" << std::endl;

	for (uint i = 0; i < gradient.size();i++)
	{
		std::cout << "CAPA " << i << std::endl;

		for (uint j = 0; j < gradient[i].size();j++)
		{
			std::cout << "coefs neurona " << j << std::endl;

			for (auto& n : gradient[i][j])
			{
				std::cout << n << " ";
			}

			std::cout << std::endl << std::endl;
		}
	}
}


