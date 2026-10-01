# Neural Network From Scratch

A C++ implementation of a multi-layer neural network trained on the MNIST handwritten digit dataset. The network achieves ~92.5% accuracy using gradient descent with dynamic programming for efficient backpropagation.

## Live drawing demo

[Try the public demo](https://neural-network-from-scratch-one.vercel.app/): draw a digit and see live predictions from the saved model. The original C++ network runs directly in your browser through WebAssembly; neural-network code and saved weights are unchanged.

To run locally: `python3 demo/server.py`, then open http://127.0.0.1:8765. See [demo setup, build, and verification](demo/README.md).

## Features

- **Architecture**: 784 → 10 → 10 (input → hidden → output)
- **Activations**: ReLU for hidden layer, Softmax for output
- **Training**: Per-sample gradient updates with an epoch-based learning-rate schedule
- **Data**: MNIST digit recognition (42,000 samples, 80/20 train/validation split)
- **Persistence**: Save/load trained models
- **Optimization**: Backpropagation using dynamic programming for efficient gradient computation

## Math implemented in the code

The equations below follow [`main.cpp`](main.cpp), including its update order. The saved model uses **784 → 10 → 10** neurons: 784 raw pixel values, 10 hidden ReLU activations, and 10 output probabilities.

### Notation and storage

Layer indices run from $0$ (input) to $L$ (output); for this model, $L=2$. Let $n_l$ be the number of neurons in layer $l$ and $K=n_L=10$.

| Symbol | Meaning | C++ storage |
| --- | --- | --- |
| $a_i^{(l)}$ | Neuron activation | `matrix[l][i].a` |
| $z_i^{(l)}$ | Weighted sum before activation | `matrix[l][i].z` |
| $b_i^{(l)}$ | Neuron bias | `matrix[l][i].bias` |
| $w_{ij}^{(l)}$ | Weight from neuron $i$ in layer $l$ to neuron $j$ in layer $l+1$ | `matrix[l][i].weights[j]` |
| $y_i$ | One-hot target for digit $i$ | `train_label_data[sample_index][i]` |

Weights are stored on the **source neuron**. `init_rand_vals` initializes weights and biases using `rand_range(-0.00001, 0.00001)`. Loading a saved model replaces the weights and the biases of all non-input layers.

### Forward propagation: `forward_prop`

The input layer receives the pixel values directly, without dividing by 255:

$$
a_i^{(0)} = x_i, \qquad x_i \in [0,255].
$$

For every connection between successive layers, the code starts with the destination bias and accumulates each incoming weighted activation:

$$
z_j^{(l+1)} = b_j^{(l+1)} + \sum_{i=0}^{n_l-1} w_{ij}^{(l)}a_i^{(l)}.
$$

Hidden layers use ReLU:

$$
a_j^{(l)} = \max(0,z_j^{(l)}), \qquad 1 \le l < L.
$$

The output layer uses softmax, evaluated through a shifted exponential sum. Let $d_{\min}$ denote the C++ constant `DBL_MIN`. These are the actual intermediate calculations in the code:

$$
m = \max\left(d_{\min}, z_0^{(L)},\ldots,z_{K-1}^{(L)}\right),
$$

$$
s = \log\left(\sum_{j=0}^{K-1}\exp(z_j^{(L)}-m)\right),
\qquad
 a_i^{(L)} = \exp(z_i^{(L)}-m-s).
$$

`DBL_MIN` is the smallest positive normalized `double`, so if all logits are negative, the initial value remains the shift. In exact arithmetic the common shift cancels, giving softmax. This describes the current initialization rather than assuming that `max_z` always equals the largest logit.

`predict` returns the digit with the largest output activation. Its comparison of `(probability, digit)` pairs selects the larger digit if probabilities tie exactly.

### Loss and output gradient: `back_prop`

The loss represented by the code is mean squared error over the $K$ output neurons:

$$
C = \frac{1}{K}\sum_{i=0}^{K-1}\left(a_i^{(L)}-y_i\right)^2.
$$

The lines that calculate the scalar `cost` are commented out, but its derivative is actively computed into `der_C_a`:

$$
g_i^{(L)} = \frac{\partial C}{\partial a_i^{(L)}}
= \frac{2}{K}\left(a_i^{(L)}-y_i\right).
$$

The nested output-layer loops apply the softmax derivative:

$$
\frac{\partial a_i^{(L)}}{\partial z_j^{(L)}}
= a_i^{(L)}\left(\mathbf{1}_{i=j}-a_j^{(L)}\right).
$$

They accumulate the signal for each output logit in `der_C_a_z[j]`:

$$
\delta_j^{(L)} = \sum_{i=0}^{K-1} g_i^{(L)}a_i^{(L)}
\left(\mathbf{1}_{i=j}-a_j^{(L)}\right).
$$

For a hidden layer, `der_ReLU` returns `input > 0`, including a derivative of zero at $z=0$:

$$
\delta_i^{(l)} = g_i^{(l)}\mathbf{1}_{z_i^{(l)}>0}.
$$

Here $g$ and $\delta$ name the backward signals stored by the implementation. At the output they are the loss derivatives above; the in-place update order below affects the signals reaching earlier layers.

### Parameter updates and their order

For each layer $l$, working backward from $L$ to $1$, the code updates its biases and incoming weights using the stored forward activations:

$$
b_j^{(l)} \leftarrow b_j^{(l)}-\eta_e\delta_j^{(l)},
$$

$$
w_{ij}^{(l-1)} \leftarrow w_{ij}^{(l-1)}
-\eta_e\delta_j^{(l)}a_i^{(l-1)}.
$$

**The weights are updated before the preceding layer's backward signal is calculated.** For $l>1$, the next signal is therefore:

$$
g_i^{(l-1)} = \sum_{j=0}^{n_l-1}\delta_j^{(l)}
\underbrace{w_{ij}^{(l-1)}}_{\text{already updated}}.
$$

The activations and pre-activations still come from the original forward pass for that sample. Consequently, the earlier-layer signals are not exactly the gradients of that loss with all weights held at their pre-update values. The equations here preserve the implementation's behavior.

### Dynamic programming in backpropagation

The backward pass reuses the loss signal already accumulated for the current layer. Each preceding neuron needs only that signal and its outgoing weights; it does not separately traverse every downstream path to the output.

The implementation uses:

- **Stored forward results:** each neuron's `a` and `z` are retained for the backward pass.
- **`der_C_a`:** a reusable vector of size `max(layer_sizes)` holding the current layer's activation signals $g$.
- **`der_C_a_z`:** a zero-initialized vector for the current layer's pre-activation signals $\delta$.

The loop follows this recurrence and overwrites `der_C_a` only after all current-layer $\delta$ values have been calculated:

```text
forward_prop(sample)                    # store all a and z

der_C_a = (2 / K) * (output - target)
for l = last layer down to 1:
    if l is the output layer:
        der_C_a_z = softmax derivative applied to der_C_a
    else:
        der_C_a_z = der_C_a * (stored z > 0)

    update this layer's biases using der_C_a_z
    update incoming weights using der_C_a_z and stored preceding activations

    if l > 1:
        clear the preceding layer's entries in der_C_a
        accumulate der_C_a from der_C_a_z and the UPDATED weights
```

For the saved **784 → 10 → 10** model, this means: calculate the 10 output signals, update the hidden-to-output weights, reuse those signals to calculate the 10 hidden signals, apply the ReLU derivative, and update the input-to-hidden weights. The code stops there; it does not calculate a gradient with respect to the input pixels.

This is the dynamic-programming part: a backward traversal with reusable layer-level results. If $E=\sum_{l=0}^{L-1}n_l n_{l+1}$ is the number of weights, the backward pass takes $O(E+K^2)$ work; the $K^2$ term comes from the explicit nested softmax-derivative loops. Its two gradient vectors require $O(\max_l n_l)$ extra space, in addition to the network's stored weights and forward values.

### Learning-rate schedule and training loop

`learning_rate_func` uses the epoch number $e$, starting at 1:

$$
\eta_e =
\begin{cases}
10^{-4}-e\,10^{-5}, & 1\le e\le10,\\
10^{-5}-(e-10)\,10^{-7}, & 11\le e\le100,\\
10^{-7}, & e>100.
\end{cases}
$$

The function ignores its `learning_rate` argument. As written, the schedule reaches zero mathematically at epoch 10, rises to $9.9\times10^{-6}$ at epoch 11, reaches $10^{-6}$ at epoch 100, and uses $10^{-7}$ afterward. These are the implemented branches, not a continuously decreasing schedule.

`fit` visits training samples in their existing order, calls `forward_prop` and `back_prop` once per sample, reports training and validation accuracy after each epoch, and saves the network after the final epoch. Updates happen per sample, without batching or shuffling. Accuracy is the fraction of samples whose predicted digit equals their label.

## Requirements

- **Compiler**: GCC 10.3.0+ (C++17 support)
- **Dependencies**: csv2 library (included)

## Quick Start

```bash
# Compile
g++ -std=c++17 -O2 main.cpp -o main

# Run (loads pre-trained model)
./main

# Training is disabled in main(); see its commented NN.fit calls.
```
