#pragma once
#include "Parameters.h"
#include <vector>

class Matrix
{
private:
	uint rows = 0;
	uint cols = 0;
	std::vector<std::vector<N_TYPE>> matrix; //vector de vectores fila
public:
	Matrix(uint rows = 0);
	Matrix(const Matrix& rhs);
	Matrix(Matrix&& rhs) noexcept;
	Matrix& operator = (const Matrix& rhs);
	Matrix& operator = (Matrix&& rhs) noexcept;

	void PlaceRow(const std::vector<N_TYPE>& row); //inserta fila
	void ColVector(const std::vector<N_TYPE>& col); //crea un vector columna
	void ShowElements(); //getters
	void ShowRC();
	Matrix operator * (const Matrix& rhs);
	std::vector<N_TYPE>& operator [](uint i);
};
