#pragma once
#include "Parameters.h"
#include <vector>

class Matrix
{
private:
	uint rows = 0;
	uint cols = 0;
	std::vector<std::vector<N_TYPE>> matrix;
public:
	Matrix(uint rows = 0);
	Matrix(const Matrix& rhs);
	Matrix(Matrix&& rhs) noexcept;
	Matrix& operator = (const Matrix& rhs);
	Matrix& operator = (Matrix&& rhs) noexcept;

	void PlaceRow(const std::vector<N_TYPE>& row);
	void ColVector(const std::vector<N_TYPE>& col);
	
	void ShowElements();
	void ShowRC();
	Matrix operator * (const Matrix& rhs);
};
