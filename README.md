# Neural Network From Scratch

**Validation accuracy: 92.68%** — 7,785 correct predictions out of 8,400 MNIST validation images.

[Live demo](https://neural-network-from-scratch-one.vercel.app/) · [Demo setup](demo/README.md) · [Source](main.cpp)

**784 inputs → 10 ReLU neurons → 10 softmax outputs**

## Math

### Notation

Layers are indexed $0,\ldots,L$, with $L=2$, layer sizes $n_l$, and $K=n_L=10$ output classes.

| Symbol | Meaning | C++ storage |
| --- | --- | --- |
| $a_i^{(l)}$ | Neuron activation | `matrix[l][i].a` |
| $z_i^{(l)}$ | Weighted sum before activation | `matrix[l][i].z` |
| $b_i^{(l)}$ | Neuron bias | `matrix[l][i].bias` |
| $w_{ij}^{(l)}$ | Weight from neuron $i$ in layer $l$ to neuron $j$ in layer $l+1$ | `matrix[l][i].weights[j]` |
| $y_i$ | One-hot target for digit $i$ | `train_label_data[sample_index][i]` |

Weights and biases are initialized with `rand_range(-0.00001, 0.00001)`.

### Forward propagation: `forward_prop`

Input:

$$
a_i^{(0)} = x_i, \qquad x_i \in [0,255].
$$

Weighted sum:

$$
z_j^{(l+1)} = b_j^{(l+1)} + \sum_{i=0}^{n_l-1} w_{ij}^{(l)}a_i^{(l)}.
$$

ReLU:

$$
a_j^{(l)} = \max(0,z_j^{(l)}), \qquad 1 \le l < L.
$$

Softmax, with $d_{\min}$ denoting `DBL_MIN`:

$$
m = \max\left(d_{\min}, z_0^{(L)},\ldots,z_{K-1}^{(L)}\right),
$$

$$
s = \log\left(\sum_{j=0}^{K-1}\exp(z_j^{(L)}-m)\right),
\qquad
 a_i^{(L)} = \exp(z_i^{(L)}-m-s).
$$

Prediction is the digit with the largest output activation.

### Loss and output gradient: `back_prop`

Mean squared error:

$$
C = \frac{1}{K}\sum_{i=0}^{K-1}\left(a_i^{(L)}-y_i\right)^2.
$$

Output gradient (`der_C_a`):

$$
g_i^{(L)} = \frac{\partial C}{\partial a_i^{(L)}}
= \frac{2}{K}\left(a_i^{(L)}-y_i\right).
$$

Softmax derivative:

$$
\frac{\partial a_i^{(L)}}{\partial z_j^{(L)}}
= a_i^{(L)}\left(\mathbf{1}_{i=j}-a_j^{(L)}\right).
$$

Output backward signal (`der_C_a_z`):

$$
\delta_j^{(L)} = \sum_{i=0}^{K-1} g_i^{(L)}a_i^{(L)}
\left(\mathbf{1}_{i=j}-a_j^{(L)}\right).
$$

Hidden-layer backward signal:

$$
\delta_i^{(l)} = g_i^{(l)}\mathbf{1}_{z_i^{(l)}>0}.
$$

### Parameter updates

For $l=L,\ldots,1$:

$$
b_j^{(l)} \leftarrow b_j^{(l)}-\eta_e\delta_j^{(l)},
$$

$$
w_{ij}^{(l-1)} \leftarrow w_{ij}^{(l-1)}
-\eta_e\delta_j^{(l)}a_i^{(l-1)}.
$$

For $l>1$, propagate using the updated weights:

$$
g_i^{(l-1)} = \sum_{j=0}^{n_l-1}\delta_j^{(l)}
\underbrace{w_{ij}^{(l-1)}}_{\text{already updated}}.
$$

### Dynamic programming in backpropagation

Each layer reuses the backward signal calculated for the layer after it:

- **Stored forward results:** each neuron's `a` and `z` are retained for the backward pass.
- **`der_C_a`:** a reusable vector of size `max(layer_sizes)` holding the current layer's activation signals $g$.
- **`der_C_a_z`:** a zero-initialized vector for the current layer's pre-activation signals $\delta$.

`der_C_a` is overwritten after the current layer’s $\delta$ values are calculated:

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

For $E=\sum_{l=0}^{L-1}n_l n_{l+1}$ weights, backpropagation takes $O(E+K^2)$ time and $O(\max_l n_l)$ extra gradient storage. The $K^2$ term is the nested softmax-derivative loop.

### Learning-rate schedule

For epoch $e$, starting at 1:

$$
\eta_e =
\begin{cases}
10^{-4}-e\,10^{-5}, & 1\le e\le10,\\
10^{-5}-(e-10)\,10^{-7}, & 11\le e\le100,\\
10^{-7}, & e>100.
\end{cases}
$$

`fit` applies one forward pass and one backward update per sample, in dataset order, for each epoch.

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
