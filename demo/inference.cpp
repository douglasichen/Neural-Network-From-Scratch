// Compile the existing implementation unchanged, replacing only its CLI entry point.
#define main original_training_main
#include "../main.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream checkpoint(argv[1]);
    uint64_t value;
    int count = 0;
    while (checkpoint >> value) ++count;
    if (!checkpoint.eof() || count != 7960) {
        std::cerr << "Expected 7960 saved parameters for 784 -> 10 -> 10.\n";
        return 2;
    }
    NeuralNetwork network;
    auto* output = std::cout.rdbuf(std::cerr.rdbuf());
    network.init({DATA_PARAM_COUNT, 10, OUTPUT_COUNT});
    network.load_network(argv[1]);
    std::cout.rdbuf(output);
    std::cout << std::setprecision(17);
    double pixels[DATA_PARAM_COUNT];
    while (std::cin >> pixels[0]) {
        for (int i = 1; i < DATA_PARAM_COUNT; ++i)
            if (!(std::cin >> pixels[i])) return 2;
        const int prediction = network.predict(pixels);
        std::cout << "{\"prediction\":" << prediction << ",\"probabilities\":[";
        for (int i = 0; i < OUTPUT_COUNT; ++i) {
            if (i) std::cout << ',';
            std::cout << network.matrix.back()[i].a;
        }
        std::cout << "]}" << std::endl;
    }
}
