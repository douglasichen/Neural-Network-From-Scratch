#include<bits/stdc++.h>
#include "csv.h"

const int LAYER_SIZES[3] = {800, 10, 10};
const int MAX_DATA_SAMPLE_COUNT 	= 10000;
const int MAX_DATA_PARAM_COUNT 		= 800;

double train_data[MAX_DATA_SAMPLE_COUNT][MAX_DATA_PARAM_COUNT];
double test_data[MAX_DATA_SAMPLE_COUNT][MAX_DATA_PARAM_COUNT];





struct NeuralNetwork {

	// math functions
	double sigmoid(double input) {
		return 1 / (1 + exp(-input));
	}

	double ReLU(double input) {
		return input > 0 ? input : 0;
	}

	double deriv_ReLU(double input) {
		return input > 0;
	}
	
	struct Neuron {
		double bias = 0;
		double output = 0;
		std::vector<double> weights;
		Neuron() {

		}
	};

	// neural network functionality
	std::vector<std::vector<Neuron>> matrix;
	double learning_rate = 0.1;
	NeuralNetwork(int layer_count, std::vector<int> layer_sizes, double learning_rate) {
		learning_rate = learning_rate;
		
		// set up matrix
		matrix.resize(layer_count);
		for (int i=0; i<layer_count; i++) {
			matrix[i].resize(layer_sizes[i]);
			if (i < layer_count-1) {
				for (int j=0; j<layer_sizes[i]; j++) {
					matrix[i][j].weights.resize(layer_sizes[i+1]);
				}
			}
		}
		
		std::cout << "Neural Network " << this << " initialized." << std::endl;
	}
};


int main() {
	std::cout << "Program Begins." << std::endl;
	NeuralNetwork NN = NeuralNetwork(3, {{10}, {10}, {10}}, 0.1);
	
	load_train_data("")
	load_test_data("");

	std::cout << "Completed Program.\n";
}