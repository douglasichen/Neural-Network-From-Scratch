// Reuse the original network, loading and prediction routines verbatim.
#define main original_training_main
#include "../main.cpp"
#undef main

namespace {
NeuralNetwork network;
double input[DATA_PARAM_COUNT];
double probabilities[OUTPUT_COUNT];
bool initialized = false;
}

extern "C" {
int initialize_model() {
    if (initialized) return 1;
    std::ifstream checkpoint("/saved_network.txt");
    uint64_t value;
    int count = 0;
    while (checkpoint >> value) ++count;
    if (!checkpoint.eof() || count != 7960) return 0;
    network.init({DATA_PARAM_COUNT, 10, OUTPUT_COUNT});
    network.load_network("/saved_network.txt");
    initialized = true;
    return 1;
}
double* input_buffer() { return input; }
double* output_buffer() { return probabilities; }
int predict_digit() {
    if (!initialized) return -1;
    int digit = network.predict(input);
    for (int i = 0; i < OUTPUT_COUNT; ++i)
        probabilities[i] = network.matrix.back()[i].a;
    return digit;
}
}
