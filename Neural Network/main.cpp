#include<bits/stdc++.h>
#include<fstream>
// #include<fmt/core.h>
#include "csv2/reader.hpp"

const int MAX_DATA_SAMPLE_COUNT 		= 42000;
const int DATA_SAMPLE_COUNT 			= 42000;

const double VALID_SAMPLE_PERCENTAGE 	= 0.2;
const int TRAIN_DATA_SAMPLE_COUNT 		= int(DATA_SAMPLE_COUNT - DATA_SAMPLE_COUNT * VALID_SAMPLE_PERCENTAGE);
const int VALID_DATA_SAMPLE_COUNT 		= DATA_SAMPLE_COUNT - TRAIN_DATA_SAMPLE_COUNT;
const int DATA_PARAM_COUNT 				= 784;
const int OUTPUT_COUNT 					= 10;

double train_data[TRAIN_DATA_SAMPLE_COUNT][DATA_PARAM_COUNT];
bool train_label_data[TRAIN_DATA_SAMPLE_COUNT][OUTPUT_COUNT];
int train_label_data_num[TRAIN_DATA_SAMPLE_COUNT];

double valid_data[VALID_DATA_SAMPLE_COUNT][DATA_PARAM_COUNT];
bool valid_label_data[VALID_DATA_SAMPLE_COUNT][OUTPUT_COUNT];
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
			// std::cout << "row: " << row_num << '\n';
			
			col_num = 0;
			for (const auto cell: row) {
				

				std::string str;
				cell.read_value(str);
				int val = stoi(str);

				if (col_num == 0) {
					
					if (row_num < TRAIN_DATA_SAMPLE_COUNT) {
						train_label_data[row_num][val] = 1;
						train_label_data_num[row_num] = val;
					}
					else {
						valid_label_data[row_num-TRAIN_DATA_SAMPLE_COUNT+1][val] = 1;
						valid_label_data_num[row_num-TRAIN_DATA_SAMPLE_COUNT+1] = val;
					}
				}
				else {
					if (row_num < TRAIN_DATA_SAMPLE_COUNT) {
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

std::string uint64_t_to_bin(uint64_t u) {
	std::string ret = "";
	for (int e=0; e<64; e++) {
		ret += (u&1 ? '1' : '0');
		if (e == 51) ret += ' ';
		u >>= 1;
	}
	std::reverse(ret.begin(), ret.end());
	return ret;
}

double uint64_t_to_double(uint64_t val) {
	double ret;
	memcpy(&ret, &val, sizeof(val));
	return ret;
}

uint64_t double_to_uint64_t(double val) {
	uint64_t ret;
	memcpy(&ret, &val, sizeof(val));
	return ret;
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
	
	double learning_rate_func(double learning_rate, int epoch) {
		if (epoch <= 10) {
			return 0.0001 - epoch*0.00001;
		}
		else if (epoch <= 100) {
			return 0.00001 - (epoch-10)*0.0000001;
		}
		return 0.0000001;

		// switch (epoch) {
		// 	case 1: return 0.0001;
		// 	case 2: return 0.00009;
		// 	case 3: return 0.00008;
		// 	case 4: return 0.00007;
		// 	case 5: return 0.00006;
		// 	case 6: return 0.00005;
		// 	case 7: return 0.00004;
		// 	case 8: return 0.00003;
		// 	case 9: return 0.00002;
		// 	case 10: return 0.00001;
			
		// }

		// return learning_rate;
		// if (epoch <= 10) return 0.0001;
		// else if (epoch <= 100) return 0.0000000001;
		// else if (epoch <= 200) return 0.00000000000001;
		// return 0.0000000000000001;

		// return learning_rate;

		// if (epoch <= 3) return 0.00011;
		// else if (epoch <= 10) return 0.0001;
		// else if (epoch <= 50) return 0.00003;
		// else return 0.00001;

		// double a = -learning_rate, b=0.1, c=32, d=learning_rate;
		// return a / (1 + exp(-b * (epoch-c))) + d;
	}

	struct Neuron {
		double bias;
		double z;
		double a;
		std::vector<double> weights;
		Neuron() {
			
		}

		void init_rand_vals() {
			double k = 0.00001;
			bias = rand_range(-k, k);
			for (double &weight : weights) {
				weight = rand_range(-k, k);
			}
		}
	};

	// neural network functionality
	int layer_count;
	std::vector<int> layer_sizes;
	std::vector<std::vector<Neuron>> matrix;
	void init(std::vector<int> _layer_sizes) {
		layer_count = _layer_sizes.size();
		layer_sizes = _layer_sizes;

		// set up matrix sizes
		matrix.resize(layer_count);
		for (int l=0; l<layer_count; l++) {
			matrix[l].resize(layer_sizes[l]);
			if (l < layer_count-1) {
				for (int i=0; i<layer_sizes[l]; i++) {
					matrix[l][i].weights.resize(layer_sizes[l+1]);
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
		
		double log_sum = 0;
		for (int i=0; i<layer_sizes[layer_count-1]; i++) {
			log_sum += exp(matrix[layer_count-1][i].z - max_z);
		}
		log_sum = log(log_sum);

		for (int i=0; i<layer_sizes[layer_count-1]; i++) {
			matrix[layer_count-1][i].a = exp(matrix[layer_count-1][i].z - max_z - log_sum);
		}


		
	}

	void back_prop(double learning_rate, int sample_index, int epoch) {
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
				matrix[l][i].bias -= der_C_a_z_b * learning_rate_func(learning_rate, epoch);
			}

			// dz / dw + learn
			for (int i=0; i<layer_sizes[l-1]; i++) {
				for (int j=0; j<layer_sizes[l]; j++) {
					double der_C_a_z_w = der_C_a_z[j] * matrix[l-1][i].a;
					matrix[l-1][i].weights[j] -= der_C_a_z_w * learning_rate_func(learning_rate, epoch);
				}
			}

			// dz / da[l-1]
			if (l > 1) {
				for (int i=0; i<layer_sizes[l-1]; i++) {
					der_C_a[i]=0;
					for (int j=0; j<layer_sizes[l]; j++) {
						der_C_a[i] += der_C_a_z[j] * matrix[l-1][i].weights[j];
					}
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
			if (predict(valid_data[sample_index]) == valid_label_data_num[sample_index]) {
				accuracy++;
			}
		}

		
		accuracy /= VALID_DATA_SAMPLE_COUNT;
		return accuracy;
	}

	void save_network(std::string filename) {
		std::ofstream file;
		file.open(filename);
		for (int l=0; l<layer_count-1; l++) {
			for (int i=0; i<layer_sizes[l]; i++) {
				for (int j=0; j<layer_sizes[l+1]; j++) {
					file << double_to_uint64_t(matrix[l][i].weights[j]) << ' ';
				}
			}
		}
		for (int l=1; l<layer_count; l++) {
			for (int i=0; i<layer_sizes[l]; i++) {
				file << double_to_uint64_t(matrix[l][i].bias) << ' ';
			}
		}
		file.close();
	}

	void load_network(std::string filename) {
		std::ifstream file;
		file.open(filename);

		uint64_t inp;
		for (int l=0; l<layer_count-1; l++) {
			for (int i=0; i<layer_sizes[l]; i++) {
				for (int j=0; j<layer_sizes[l+1]; j++) {
					file >> inp;
					matrix[l][i].weights[j] = uint64_t_to_double(inp);
				}
			}
		}
		for (int l=1; l<layer_count; l++) {
			for (int i=0; i<layer_sizes[l]; i++) {
				file >> inp;
				matrix[l][i].bias = uint64_t_to_double(inp);
			}
		}
		file.close();
	}

	void fit(double learning_rate, int epochs, std::string saved_network_filename) {
		for (int epoch=1; epoch<=epochs; epoch++) {
			for (int sample_index=0; sample_index<TRAIN_DATA_SAMPLE_COUNT; sample_index++) {
				
				forward_prop(train_data[sample_index]);
				back_prop(learning_rate, sample_index, epoch);
				
				// for (int l=1; l<layer_count; l++) {
					// for (int i=0; i<layer_sizes[l]; i++) {
					// 	std::cout << matrix[l][i].a << ' ';
					// }
					// std::cout<<'\n';
				// }

				// reset z,a for next run. Note: forward_prop resets it.
				// for (int l=0; l<layer_count; l++) {
				// 	for (int i=0; i<layer_sizes[l]; i++) {
				// 		matrix[l][i].z = matrix[l][i].a = 0;
				// 	}
				// }
			}
			// finished epoch	
			std::cout << "Epoch " << epoch 
					<< ": train_accuracy=" << train_accuracy()
					<< ", valid_accuracy=" << valid_accuracy()
					<< std::endl;	
		}

		// for (int l=1; l<layer_count; l++) {
		// 	for (int i=0; i<layer_sizes[l]; i++) {
		// 		std::cout << matrix[l][i].bias << ' ';
		// 	}
		// 	std::cout<<'\n';
		// }

		save_network(saved_network_filename);
	}

};

int main() {
	std::cout << std::fixed << std::setprecision(4) << "Program Begins." << std::endl;
	init_helpers();
	NeuralNetwork NN;

	load_data("digit_recognizer/train.csv");
	
	// NN.init({DATA_PARAM_COUNT, 10, OUTPUT_COUNT});
	// NN.fit(0.001, 100, "saved_network.txt"); 
	
	// for (int n=0; n<10; n++) {
	// 	NN.init({DATA_PARAM_COUNT, 10, OUTPUT_COUNT});
	// 	NN.fit(0.1, 10, "saved_network_" + std::to_string(n) + ".txt");
	// }

	NN.init({DATA_PARAM_COUNT, 10, OUTPUT_COUNT});
	// NN.fit(0.00001, 10, "saved_network.txt");
	NN.load_network("saved_network.txt");
	std::cout << "train_accuracy=" << NN.train_accuracy()
				<< ", valid_accuracy=" << NN.valid_accuracy()
				<< std::endl;

	std::cout << "Completed Program.\n";
}