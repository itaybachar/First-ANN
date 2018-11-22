#pragma once

#include <vector>
#include <iostream>

class Matrix {
public:
	Matrix();
	Matrix(int height, int width);
	Matrix(std::vector<std::vector<double>> const &mat);

	Matrix add(Matrix const &m) const;
	Matrix subtract(Matrix const &m) const;
	Matrix multiply(double const &val);
	Matrix multiply(Matrix const &m) const;
	Matrix dot(Matrix const &m) const;
	Matrix applyFunction(double(*function)(double)) const;
	Matrix transpose() const;
	Matrix normalize(double min_in, double max_in, double min_out, double max_out);
	std::vector<std::vector<double> > cloneArray();
	void print(std::ostream &flux) const;
	std::vector<std::vector<double> >* getMatrix() {
		return &matrix;
	}
	int greatestValIndex();
private:
	std::vector<std::vector<double> > matrix;
	int height;
	int width;
};
std::ostream& operator<<(std::ostream& flux, Matrix const &m);


