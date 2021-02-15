#include "Matrix.h"
#include <iostream>

Matrix::Matrix(uint rows)
{
	if (rows)
		matrix.reserve(rows);
}

Matrix::Matrix(const Matrix& rhs)
{
	*this = rhs;
}

Matrix::Matrix(Matrix&& rhs) noexcept
{
	*this = std::move(rhs);
}

Matrix& Matrix::operator=(const Matrix& rhs)
{
	if (this != &rhs)
	{
		rows = rhs.rows;
		cols = rhs.cols;
		matrix = rhs.matrix;
	}

	return *this;
}

Matrix& Matrix::operator=(Matrix&& rhs) noexcept
{
	if (this != &rhs)
	{
		rows = rhs.rows;
		cols = rhs.cols;
		matrix = rhs.matrix;
	}

	return *this;
}

void Matrix::PlaceRow(const std::vector<N_TYPE>& row)
{
	rows++;
	cols = row.size();
	matrix.emplace_back(row);
}

void Matrix::ColVector(const std::vector<N_TYPE>& col)
{
	cols = 1;
	rows = col.size();
	matrix.clear();
	matrix.reserve(col.size());

	for (uint i = 0; i < col.size(); i++)
		matrix.emplace_back(std::vector < N_TYPE>(1, col[i]));
}

Matrix Matrix::operator*(const Matrix& rhs)
{
	if (cols != rhs.rows)
	{
		std::cout << "dimenisones para multiplicar incorrectas" << std::endl;
		return{};
	}

	Matrix output(rows);

	for (uint i = 0; i < rows; i++)
	{
		output.matrix.emplace_back(); //crear el vector fila
		output.matrix[i].reserve(rhs.cols); //reservar espacios

		for (uint j = 0; j < rhs.cols; j++)
		{
			N_TYPE result = 0;

			for (uint n = 0; n < cols; n++)
				result += matrix[i][n] * rhs.matrix[n][j];

			output.matrix[i].emplace_back(result);
		}
	}

	output.rows = rows;
	output.cols = rhs.cols;
	return output;
}

std::vector<N_TYPE>& Matrix::operator[](uint i)
{
	if (i >= rows || i < 0)
	{
		std::cout << "fuera de rango, se usará la última fila" << std::endl;
		return matrix[(size_t)rows - 1];
	}
	else
		return matrix[(size_t)i];
}

void Matrix::ShowElements()
{
	std::cout << "la matriz es: " << std::endl;

	for (auto& rows : matrix)
	{
		for (auto& cols : rows)
			std::cout << cols << " ";

		std::cout << std::endl;
	}

}

void Matrix::ShowRC()
{
	std::cout << "hay " << rows << " filas y " << cols << " columnas" << std::endl;
}


