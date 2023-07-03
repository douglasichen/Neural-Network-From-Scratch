#include<bits/stdc++.h>
#include "csv2/reader.hpp"

const int TRAIN_DATA_SAMPLE_COUNT 	= 4200;
const int TEST_DATA_SAMPLE_COUNT 	= 2800;
const int DATA_PARAM_COUNT 			= 784;

double train_data[TRAIN_DATA_SAMPLE_COUNT][DATA_PARAM_COUNT];
double train_label_data[TRAIN_DATA_SAMPLE_COUNT][10];

double test_data[TEST_DATA_SAMPLE_COUNT][DATA_PARAM_COUNT];
double test_label_data[TEST_DATA_SAMPLE_COUNT][10];

void init_helpers() {
	srand(time(NULL));
}

void load_train_data(std::string dirname) {
	// init train_label_data
	memset(train_label_data, 0, sizeof(train_label_data));

	csv2::Reader<csv2::delimiter<','>, 
               csv2::quote_character<'"'>, 
               csv2::first_row_is_header<true>,
               csv2::trim_policy::trim_whitespace> csv;
               
	if (csv.mmap(dirname)) {
		// const auto header = csv.header();

		int row_num = 0, col_num = 0;
		for (const auto row: csv) {
			if (row_num+1 > TRAIN_DATA_SAMPLE_COUNT) break;

			col_num = 0;
			for (const auto cell: row) {
				std::string str;
				cell.read_value(str);
				int val = stoi(str);

				if (col_num == 0) {
					train_label_data[row_num][val] = 1;
				}
				else {
					train_data[row_num][col_num-1] = val;
				}

				col_num++;
			}
			row_num++;
		}
	}

	std::cout << "Training Data Loaded." << std::endl;
}

void load_test_data(std::string dirname) {
	// init train_label_data
	memset(test_label_data, 0, sizeof(test_label_data));

	csv2::Reader<csv2::delimiter<','>, 
               csv2::quote_character<'"'>, 
               csv2::first_row_is_header<true>,
               csv2::trim_policy::trim_whitespace> csv;
               
	if (csv.mmap(dirname)) {
		// const auto header = csv.header();

		int row_num = 0, col_num = 0;
		for (const auto row: csv) {
			if (row_num+1 > TEST_DATA_SAMPLE_COUNT) break;

			col_num = 0;
			for (const auto cell: row) {
				std::string str;
				cell.read_value(str);
				int val = stoi(str);
				
				if (col_num == 0) {
					test_label_data[row_num][val] = 1;
				}
				else {
					test_data[row_num][col_num-1] = val;
				}

				col_num++;
			}
			row_num++;
		}
	}

	std::cout << "Test Data Loaded." << std::endl;
}

double rand_range(double fMin, double fMax) {
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

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
		double bias;
		double z;
		double a;
		std::vector<double> weights;
		Neuron() {
			
		}

		void init_rand_vals() {
			bias = rand_range(-0.01, 0.01);
			for (double &weight : weights) {
				weight = rand_range(-0.01, 0.01);
			}
		}
	};

	// neural network functionality
	int layer_count;
	std::vector<int> layer_sizes;
	std::vector<std::vector<Neuron>> matrix;
	NeuralNetwork(int _layer_count, std::vector<int> _layer_sizes) {
		layer_count = _layer_count;
		layer_sizes = _layer_sizes;

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

		// initialize random neurons
		for (int l=0; l<layer_count; l++) {
			for (int i=0; i<layer_sizes[l]; i++) {
				matrix[l][i].init_rand_vals();
			}
		}
		
		std::cout << "Neural Network " << this << " initialized." << std::endl;
	}

	double train_accuracy() {
		return 0;
	}

	double test_accuracy() {
		return 0;
	}

	void fit(double learning_rate, int epochs) {
		for (int epoch=1; epoch<=epochs; epoch++) {
			for (int sample_index=0; sample_index<1; sample_index++) {

				// FORWARDS
				// init first layer with data
				for (int i=0; i<layer_sizes[0]; i++) {
					matrix[0][i].a = train_data[sample_index][i];
				}

				// forwards propagation
				/*
					Every layer activates with ReLU
					EXCEPT last layer that activates with SoftMax
				*/
				for (int l=0; l<layer_count-1; l++) {
					for (Neuron neuron : matrix[l]) {
						for (int i=0; i<neuron.weights.size(); i++) {
							matrix[l+1][i].z += neuron.weights[i]*neuron.a + matrix[l+1][i].bias;
						}
					}

					if (l < layer_count-2) {
						for (int i=0; i<layer_sizes[l+1]; i++) {
							matrix[l+1][i].a = ReLU(matrix[l+1][i].z);
						}
					}
				}

				// add softmax activation
				double max_z = DBL_MIN;
				for (int i=0; i<layer_sizes[layer_count-1]; i++) {
					max_z = std::max(max_z, matrix[layer_count-1][i].z);
				}
				
				double sum = 0;
				for (int i=0; i<layer_sizes[layer_count-1]; i++) {
					sum += exp(matrix[layer_count-1][i].z - max_z);
				}

				for (int i=0; i<layer_sizes[layer_count-1]; i++) {
					matrix[layer_count-1][i].a = exp(matrix[layer_count-1][i].z - max_z - log(sum));
				}

				double cost = 0;
				for (int i=0; i<layer_sizes[layer_count-1]; i++) {
					cost += pow(matrix[layer_count-1][i].a - train_label_data[sample_index][i], 2);
				}
				cost /= layer_sizes[layer_count-1];
				
				// 

				// BACKWARDS
				// C = 1/m * sum[ (a - y)^2 ]
				// a = softmax(z)
				// z = sum

				// init dC/da
				// A = dC/da
				// B = da/dz
				// C = dz/dw
				// D = dz/db
				double A, B, C, D;

				// reset for next run
				for (int l=0; l<layer_count; l++) {
					for (int i=0; i<layer_sizes[l]; i++) {
						matrix[l][i].z = matrix[l][i].a = 0;
					}
				}
			}
			// finished epoch	
			std::cout << "Epoch " << epoch 
					<< ": train_accuracy=" << train_accuracy()
					<< ", test_accuracy=" << test_accuracy() 
					<< std::endl;
		}
	}

};


int main() {
	std::cout << std::fixed << std::setprecision(2) << "Program Begins." << std::endl;
	init_helpers();
	NeuralNetwork NN = NeuralNetwork(3, {{DATA_PARAM_COUNT}, {10}, {10}});
	
	load_train_data("digit_recognizer/train.csv");
	load_test_data("digit_recognizer/test.csv");

	NN.fit(0.001, 2);

	std::cout << "Completed Program.\n";
}