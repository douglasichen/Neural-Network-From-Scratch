#include<bits/stdc++.h>
#include "csv2/reader.hpp"

const int MAX_DATA_SAMPLE_COUNT = 42000;
const int DATA_SAMPLE_COUNT = 20000;

const double VALID_SAMPLE_PERCENTAGE = 0.2;
const int TRAIN_DATA_SAMPLE_COUNT = int(DATA_SAMPLE_COUNT - DATA_SAMPLE_COUNT * VALID_SAMPLE_PERCENTAGE);
const int VALID_DATA_SAMPLE_COUNT = int(DATA_SAMPLE_COUNT * VALID_SAMPLE_PERCENTAGE);

// const int TRAIN_DATA_SAMPLE_COUNT 	= 4200;
// const int TEST_DATA_SAMPLE_COUNT 	= 2800;
const int DATA_PARAM_COUNT 			= 784;

double train_data[TRAIN_DATA_SAMPLE_COUNT][DATA_PARAM_COUNT];
bool train_label_data[TRAIN_DATA_SAMPLE_COUNT][10];
int train_label_data_num[TRAIN_DATA_SAMPLE_COUNT];

double valid_data[VALID_DATA_SAMPLE_COUNT][DATA_PARAM_COUNT];
bool valid_label_data[VALID_DATA_SAMPLE_COUNT][10];
int valid_label_data_num[VALID_DATA_SAMPLE_COUNT];

void init_helpers() {
	srand(time(NULL));
}

void load_data(std::string dirname) {
	// init train_label_data
	memset(train_label_data, 0, sizeof(train_label_data));
	memset(valid_label_data, 0, sizeof(valid_label_data));

	csv2::Reader<csv2::delimiter<','>, 
               csv2::quote_character<'"'>, 
               csv2::first_row_is_header<true>,
               csv2::trim_policy::trim_whitespace> csv;
               
	if (csv.mmap(dirname)) {
		// const auto header = csv.header();

		int row_num = 0, col_num = 0;
		for (const auto row: csv) {
			if (row_num+1 > DATA_SAMPLE_COUNT) break;
			
			col_num = 0;
			for (const auto cell: row) {
				std::string str;
				cell.read_value(str);
				int val = stoi(str);

				if (col_num == 0) {
					
					if (row_num+1 <= TRAIN_DATA_SAMPLE_COUNT) {
						train_label_data[row_num][val] = 1;
						train_label_data_num[row_num] = val;
					}
					else {
						valid_label_data[row_num-TRAIN_DATA_SAMPLE_COUNT+1][val] = 1;
						valid_label_data_num[row_num-TRAIN_DATA_SAMPLE_COUNT+1] = val;
					}
				}
				else {
					if (row_num+1 <= TRAIN_DATA_SAMPLE_COUNT) {
						train_data[row_num][col_num-1] = val;
					}
					else {
						valid_data[row_num-TRAIN_DATA_SAMPLE_COUNT+1][col_num-1] = val;
					}
				}
				col_num++;
			}
			row_num++;
		}
	}

	std::cout << "Data Loaded." << std::endl;
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

	double der_ReLU(double input) {
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
			bias = rand_range(-0.02, 0.02);
			for (double &weight : weights) {
				weight = rand_range(-0.02, 0.02);
			}
		}
	};

	struct BP_node {
		int layer, i, j;
		double A, B;
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

	void forward_prop(double sample[]) {
		// FORWARDS
		// init first layer with data
		for (int i=0; i<layer_sizes[0]; i++) {
			matrix[0][i].a = sample[i];
		}

		// forwards propagation
		/*
			Every layer activates with ReLU
			EXCEPT last layer that activates with SoftMax
		*/
		for (int l=0; l<layer_count-1; l++) {
			// add biases
			for (int i=0; i<layer_sizes[l+1]; i++) {
				matrix[l+1][i].z = matrix[l+1][i].bias;
			}

			// add weight * a
			for (Neuron neuron : matrix[l]) {
				for (int i=0; i<neuron.weights.size(); i++) {
					matrix[l+1][i].z += neuron.weights[i]*neuron.a;
				}
			}

			// add ReLU activation
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
	}

	void back_prop(double learning_rate, int sample_index) {
		// double cost = 0;
		// for (int i=0; i<layer_sizes[layer_count-1]; i++) {
		// 	cost += pow(matrix[layer_count-1][i].a - train_label_data[sample_index][i], 2.0);
		// }
		// cost /= layer_sizes[layer_count-1];

		// dC / da
		std::vector<double> der_C_a(*max_element(layer_sizes.begin(), layer_sizes.end()));
		for (int i=0; i<layer_sizes[layer_count-1]; i++) {
			der_C_a[i] = 2.0 / layer_sizes[layer_count-1] * 
						(matrix[layer_count-1][i].a - train_label_data[sample_index][i]);
		}

		for (int l=layer_count-1; l>0; l--) {
			// da / dz
			std::vector<double> der_C_a_z(layer_sizes[l]);
			if (l == layer_count-1) {
				for (int i=0; i<layer_sizes[l]; i++) {
					for (int j=0; j<layer_sizes[l]; j++) {
						der_C_a_z[j] += der_C_a[i] * matrix[l][i].a * ((i==j) - matrix[l][j].a);
					}
				}
			}
			else {
				for (int i=0; i<layer_sizes[l]; i++) {
					der_C_a_z[i] = der_C_a[i] * der_ReLU(matrix[l][i].z);
				}
			}

			// dz / db + learn
			for (int i=0; i<layer_sizes[l]; i++) {\
				double der_C_a_z_b = der_C_a_z[i];
				matrix[l][i].bias -= der_C_a_z_b * learning_rate;
			}

			// dz / dw + learn
			for (int i=0; i<layer_sizes[l-1]; i++) {
				for (int j=0; j<layer_sizes[l]; j++) {
					double der_C_a_z_w = der_C_a_z[j] * matrix[l-1][i].a;
					matrix[l-1][i].weights[j] -= der_C_a_z_w * learning_rate;
				}
			}

			// dz / da[l-1]
			for (int i=0; i<layer_sizes[l-1]; i++) {
				der_C_a[i]=0;
				for (int j=0; j<layer_sizes[l]; j++) {
					der_C_a[i] += der_C_a_z[j] * matrix[l-1][i].weights[j];
				}
			}
		}
	}

	int predict(double sample[]) {
		forward_prop(sample);
		std::pair<double,int> max_arg={0,0};
		for (int i=0; i<layer_sizes[layer_count-1]; i++) {
			max_arg = max(max_arg, {matrix[layer_count-1][i].a, i});
		}
		return max_arg.second;
	}

	double train_accuracy() {
		double accuracy = 0;
		for (int sample_index=0; sample_index<TRAIN_DATA_SAMPLE_COUNT; sample_index++) {
			if (predict(train_data[sample_index]) == train_label_data_num[sample_index]) {
				accuracy++;
			}
		}
		accuracy /= TRAIN_DATA_SAMPLE_COUNT;
		return accuracy;
	}

	double valid_accuracy() {
		double accuracy = 0;
		for (int sample_index=0; sample_index<VALID_DATA_SAMPLE_COUNT; sample_index++) {
			int prediction = predict(valid_data[sample_index]);
			if (prediction == valid_label_data_num[sample_index]) {
				accuracy++;
			}
		}
		accuracy /= VALID_DATA_SAMPLE_COUNT;
		return accuracy;
	}

	void fit(double learning_rate, int epochs) {
		for (int epoch=1; epoch<=epochs; epoch++) {
			for (int sample_index=0; sample_index<TRAIN_DATA_SAMPLE_COUNT; sample_index++) {
				
				forward_prop(train_data[sample_index]);
				back_prop(learning_rate, sample_index);
				
				// reset z,a for next run
				for (int l=0; l<layer_count; l++) {
					for (int i=0; i<layer_sizes[l]; i++) {
						matrix[l][i].z = matrix[l][i].a = 0;
					}
				}
			}
			// finished epoch	
			std::cout << "Epoch " << epoch 
					<< ": train_accuracy=" << train_accuracy()
					<< ", valid_accuracy=" << valid_accuracy() 
					<< std::endl;
		}
	}

};


int main() {
	std::cout << std::fixed << std::setprecision(4) << "Program Begins." << std::endl;
	init_helpers();
	NeuralNetwork NN = NeuralNetwork(3, {{DATA_PARAM_COUNT}, {10}, {10}});
	
	load_data("digit_recognizer/train.csv");

	NN.fit(0.0001, 20);

	std::cout << "Completed Program.\n";
}