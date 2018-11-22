#include "Matrix.h"

#include <assert.h>
#include <sstream>
#include <math.h>

Matrix::Matrix(){}

Matrix::Matrix(int height, int width)
{
	this->matrix = std::vector<std::vector<double>>(height, std::vector<double>(width));
	this->width = width;
	this->height = height;
}

Matrix::Matrix(std::vector<std::vector<double>> const & mat)
{
	this->matrix = mat;
	this->height = mat.size();
	this->width = mat[0].size();
}

Matrix Matrix::add(Matrix const & m) const
{
	assert(height == m.height && width == m.width);

	Matrix res(height, width);

	for (int i = 0; i < height;  i++) {
		for (int j = 0; j < width; j++) {
			res.matrix[i][j] = matrix[i][j] + m.matrix[i][j];
		}
	}

	return res;
}

Matrix Matrix::subtract(Matrix const & m) const
{
	assert(height == m.height && width == m.width);

	Matrix res(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			res.matrix[i][j] = matrix[i][j] - m.matrix[i][j];
		}
	}

	return res;
}

Matrix Matrix::multiply(double const & val)
{
	Matrix res(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			res.matrix[i][j] = matrix[i][j] * val;
		}
	}

	return res;
}

Matrix Matrix::multiply(Matrix const & m) const
{
	assert(height == m.height && width == m.width);

	Matrix res(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			res.matrix[i][j] = matrix[i][j] * m.matrix[i][j];
		}
	}

	return res;
}

Matrix Matrix::dot(Matrix const & m) const
{
	assert(width == m.height);

	Matrix res(height, m.width);

	double w = 0;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < m.width; j++) {
			for (int h = 0; h < width; h++) {
				w += matrix[i][h]*m.matrix[h][j];
			}
			res.matrix[i][j] = w;
			w = 0;
		}
	}
	return res;
}

Matrix Matrix::applyFunction(double(*function)(double)) const
{
	Matrix res(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			res.matrix[i][j] = (*function)(matrix[i][j]);
		}
	}
	
	return res;
}

int Matrix::greatestValIndex() {
	int bestIndex = -1;
	int bestVal = -10000;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (bestVal < matrix[i][j]) {
				bestVal = matrix[i][j];
				bestIndex = j;
			}
		}
	}
	return bestIndex;
}

Matrix Matrix::transpose() const
{
	Matrix res(width, height);

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			res.matrix[i][j] = matrix[j][i];
		}
	}
	
	return res;
}

void Matrix::print(std::ostream &flux) const
{
	int i, j;
	std::stringstream ss;


	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			flux << matrix[i][j]<< " ";
		}
		flux << std::endl;
	}
}


Matrix Matrix::normalize(double min_in, double max_in,double min_out,double max_out)
{
	Matrix res(height, width);
	double ratio = min_in / max_in;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double x = matrix[i][j] / (max_in - min_in);
			res.matrix[i][j] = min_out + (max_out - min_out) * x;
		}
	}
	return res;
}

std::vector<std::vector<double>> Matrix::cloneArray()
{
	std::vector<std::vector<double> > res(height, std::vector<double>(width));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			res[i][j] = matrix[i][j];
		}
	}
	return res;
}


std::ostream & operator<<(std::ostream & flux, Matrix const & m)
{
	m.print(flux);
	return flux;
}
