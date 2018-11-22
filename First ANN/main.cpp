#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iomanip>

#include "Matrix.h"
#include "MNISTReader.h"



Matrix X, W1, B1, H, W2, B2, Y, dJdB1, dJdB2, dJdW1, dJdW2;
double learningRate;

//Random Value for wieghts and biases function
double random(double x) {
	return (double)(rand() % 10000 + 1) / 10000 - 0.5;
}

//Activation for Forward Propagation
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

//Activation for Back Propagation(Derivative)
double sigmoidPrime(double x) {
	return exp(-x) / pow((1.0 + exp(-x)), 2.0);
}

//Round off values function
double stepFunction(double x) {
	if (x > 0.9) {
		return 1.0;
	}
	else if (x < 0.1) {
		return 0.0;
	} else return x;
}

//Initialize Neural Network
void initNet(int inputNeurons, int hiddenNeurons, int outputNeurons, double rate) {
	learningRate = rate;

	W1 = Matrix(inputNeurons, hiddenNeurons);
	W2 = Matrix(hiddenNeurons, outputNeurons);
	B1 = Matrix(1, hiddenNeurons);
	B2 = Matrix(1, outputNeurons);

	W1 = W1.applyFunction(random);
	W2 = W2.applyFunction(random);
	B1 = B1.applyFunction(random);
	B2 = B2.applyFunction(random);
}

//Forward Propagation
Matrix computeOutput(std::vector<double> input) {
	//Normalize inputs between 0 and 1
	X = Matrix({ input }).normalize(0,255,0,1);
	
	//Forward Propagation
	H = X.dot(W1).add(B1).applyFunction(sigmoid);
	Y = H.dot(W2).add(B2).applyFunction(sigmoid);
	return Y;
}

//Back Propagation
void learn(std::vector<double> expectedOut) {
	Matrix Y2({ expectedOut });

	//Calculating Gradients using squared error loss function.
	dJdB2 = Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidPrime));
	dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidPrime));
	dJdW2 = H.transpose().dot(dJdB2);
	dJdW1 = X.transpose().dot(dJdB1);

	//Updating weights and biases
	W1 = W1.subtract(dJdW1.multiply(learningRate));
	W2 = W2.subtract(dJdW2.multiply(learningRate));
	B1 = B1.subtract(dJdB1.multiply(learningRate));
	B2 = B2.subtract(dJdB2.multiply(learningRate));
}

//Save Net to a file
void saveValues(const char* filename) {
	std::ofstream file(filename);
	
	//W1
	file << (*W1.getMatrix()).size() << " " << (*W1.getMatrix())[0].size() << '\n';
	for (int i = 0; i < (*W1.getMatrix()).size(); i++) {
		for (int j = 0; j < (*W1.getMatrix())[0].size(); j++) {
			file << (*W1.getMatrix())[i][j] << '\n';
		}
	}

	//W2
	file << (*W2.getMatrix()).size() << " " << (*W2.getMatrix())[0].size() << '\n';
	for (int i = 0; i < (*W2.getMatrix()).size(); i++) {
		for (int j = 0; j < (*W2.getMatrix())[0].size(); j++) {
			file << (*W2.getMatrix())[i][j] << '\n';
		}
	}

	//B1
	file << (*B1.getMatrix()).size() << " " << (*B1.getMatrix())[0].size() << '\n';
	for (int i = 0; i < (*B1.getMatrix()).size(); i++) {
		for (int j = 0; j < (*B1.getMatrix())[0].size(); j++) {
			file << (*B1.getMatrix())[i][j] << '\n';
		}
	}

	//B2
	file << (*B2.getMatrix()).size() << " " << (*B2.getMatrix())[0].size() << '\n';
	for (int i = 0; i < (*B2.getMatrix()).size(); i++) {
		for (int j = 0; j < (*B2.getMatrix())[0].size(); j++) {
			file << (*B2.getMatrix())[i][j] << '\n';
		}
	}

	file.close();

}

//Load a Net from a file
void loadNet(const char* filename) {
	std::ifstream file(filename);

	std::string line;
	int height;
	int width;
	double val;

	//W1
	file >> height;
	file >> width;
	W1 = Matrix(height, width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			file >> val;
			(*W1.getMatrix())[i][j] = val;
	
		}
	}

	//W2
	file >> height;
	file >> width;
	W2 = Matrix(height, width);
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			file >> val;
			(*W2.getMatrix())[i][j] = val;
		}
	}

	//B1
	file >> height;
	file >> width;
	B1 = Matrix(height, width);

	for (int i = 0; i < height; i++) {
		
		for (int j = 0; j < width; j++) {
			file >> val;
			(*B1.getMatrix())[i][j] = val;
		}
	}

	//B2
	file >> height;
	file >> width;
	B2 = Matrix(height, width);

	for (int i = 0; i < height; i++) {
		
		for (int j = 0; j < width; j++) {
			file >> val;
			(*B2.getMatrix())[i][j] = val;
		}
	}
	file.close();
}

//Train a net
void train(const char* saveFileName, std::vector<std::vector<double>> input, std::vector<std::vector<double>> output,int iterations) {
	//Initialize network
	initNet(input[0].size(), 14, output[0].size(), 0.2);

	//Calculate the learning rate decay
	double decay = learningRate / iterations;
	
	std::cout << "#0/" << iterations << std::endl;
		
	//Itereting through the inputs and learining, this is done n amount of times to refine network
	for (int i = 0; i < iterations; i++) {
		learningRate = learningRate * 1 / (1 + decay * i);
		for (int j = 0; j < input.size(); j++) {
			computeOutput(input[j]);
			learn(output[j]);
		}
		std::cout << "#" << i + 1 << "/" << iterations << std::endl;
	}
	
	//Save the network
	saveValues(saveFileName);
}

//Test a net
void test(const char* filename, std::vector<std::vector<double>> input, std::vector<std::vector<double>> output) {
	//Load network
	loadNet(filename);

	int calc, expected, correct = 0;

	std::cout << "\n Expected:Computed" << std::endl;

	//Iterate through inputs and calculating the output, then displaying how accurate the network is
	for (int i = 0; i < input.size(); i++) {
		for (int j = 0; j < 10; j++) {
			if (output[i][j] == 1) {
				std::cout << j;
				expected = j;
				break;
			}
		}
		calc = computeOutput(input[i]).applyFunction(stepFunction).greatestValIndex();
		if (calc == expected) correct++;

		std::cout << std::setprecision(2) << std::fixed;

		std::cout << ":" << calc << "| " << (100 * (correct) / (double)i) << "% Correct" << std::endl;
	}
}

int main() {
	srand(time(NULL));

	std::vector<std::vector<double>> input, output;
	std::vector<double>readOut;
	MNISTReader reader;

	//Loading Learn
	reader.loadImages("train-images.idx3-ubyte",input);
	reader.loadLabels("train-labels.idx1-ubyte", readOut);

	//Loading Test
	//reader.loadImages("t10k-images.idx3-ubyte", input);
	//reader.loadLabels("t10k-labels.idx1-ubyte", readOut);
	

	output = std::vector<std::vector<double>>();
	output.resize(readOut.size());

	for (int i = 0; i < readOut.size();i++) {
		output[i].resize(10);
		output[i][(int)readOut[i]] = 1;
	}

	std::cout << "File Loaded!" << std::endl;


	train("test.txt",input, output,1);
	//test("numbersTrained.txt",input, output);
	
	system("pause");
}
 