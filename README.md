# Neural Network From Scratch

A C++ implementation of a multi-layer neural network trained on the MNIST handwritten digit dataset. The network achieves ~92.5% accuracy using gradient descent with dynamic programming for efficient backpropagation.

## Features

- **Architecture**: 784 → 10 → 10 (input → hidden → output)
- **Activations**: ReLU for hidden layer, Softmax for output
- **Training**: Stochastic gradient descent with adaptive learning rate
- **Data**: MNIST digit recognition (42,000 samples, 80/20 train/validation split)
- **Persistence**: Save/load trained models
- **Optimization**: Backpropagation using dynamic programming for efficient gradient computation

## Algorithm Details

### Training Process
1. **Forward Pass**: Data flows through network layers with ReLU/Softmax activations
2. **Backpropagation**: Uses dynamic programming to efficiently compute gradients layer by layer
3. **Gradient Descent**: Updates weights and biases to minimize cost function
4. **Adaptive Learning Rate**: Decreases learning rate over epochs for better convergence

### Network Architecture
- **Input Layer**: 784 neurons (28×28 pixel MNIST images)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (digit classes 0-9)

## Requirements

- **Compiler**: GCC 10.3.0+ (C++17 support)
- **Dependencies**: csv2 library (included)

## Quick Start

```bash
# Compile
g++ main.cpp

# Run (loads pre-trained model)
./main

# Train new model (uncomment lines 441-442 in main.cpp)
```
