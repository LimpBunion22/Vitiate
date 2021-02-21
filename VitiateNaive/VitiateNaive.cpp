// VitiateNaive.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include "RED.h"
#include "Matrix.h"

class Prueba
{
public:
	Prueba()
	{
		std::cout << "Prueba default" << std::endl;
	}
	Prueba(const Prueba& rhs)
	{
		std::cout << "Prueba copia" << std::endl;
	}
	Prueba& operator = (const Prueba& rhs)
	{
		std::cout << "Prueba assign" << std::endl;
		return *this;
	}
	~Prueba()
	{
		std::cout << "Destruyendo" << std::endl;
	}
};

int main()
{
	// Prueba* prueba = reinterpret_cast<Prueba*>(operator new(sizeof(Prueba) * 5));
	// for (int i = 0; i < 5; i++)
	// {
	// 	new(&prueba[i]) Prueba();
	// }

	// for (int i = 0; i < 5; i++)
	// {
	// 	prueba[i].~Prueba();
	// }

	// operator delete (prueba);

	// std::vector<Prueba> vec1(5);
	// std::cout << "nuevo vec" << std::endl;
	// std::vector<Prueba> vec2(8);
	// std::cout << "copiando" << std::endl;
	// vec1 = vec2;
	// std::cout << "copiado" << std::endl;

	// uint neurons[] = { 2,2 };
	// uint layers = sizeof(neurons) / sizeof(neurons[0]);
	// RED red(1, neurons, layers);
	// std::vector<N_TYPE> outs = red.Forward({4});

	// std::cout << "salidas finales" << std::endl;

	// for (auto i : outs)
	// 	std::cout << i << " " << std::endl;
	
	// red.PrintCoefs();

	std::vector<N_TYPE> neurons = { 5,5,5,5,5};
	std::vector<N_TYPE> inputs = { -2,1,10,-1};
	std::vector<N_TYPE> outputs ={ 4,-1,1,1,2 };
	RED red(inputs, outputs, neurons);
	std::cout << "COEFICIENTES" << std::endl;
	red.PrintCoefs();
	std::vector<N_TYPE> outs = red.Forward({ -2,1,10,-1 });
	std::cout << "SALIDAS FINALES" << std::endl;

	for (auto i : outs)
		std::cout << i << " ";

	std::cout << std::endl << std::endl;
	red.PrintGradient(red.Gradient());
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
