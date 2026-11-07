# Sho3la: a neural network library from scratch in C++

**Tensors, dense layers and activations written by hand, without any ML framework, to understand what PyTorch
does under the hood.** (*Sho3la* is Algerian Arabic for "flame", a nod to *Torch*.)

## What is built

| Part | What it does |
|---|---|
| `Sho3la` tensor | N-dimensional shape, element-wise `+` / `-` (with `+=` / `-=`), matrix product `dot`, `sum` along an axis with `keepDims`, broadcasting a bias over a batch |
| `Sho3laDenseLayer` | fully connected layer: `output = input · W + b` over a batch |
| Activation layers | ReLU and Softmax, behind a common activation interface |

```cpp
Sho3laDenseLayer dense(4, 2);           // 4 inputs, 2 neurons
dense.setWeights(Sho3la({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
dense.setBias(Sho3la({1, 2}, {1, 1}));

Sho3la batch({3, 4}, {1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12});
dense.forward(batch);
dense.getOutput().print();              // same result as batch.dot(W) + b
```

## Build

```bash
cmake -S . -B build
cmake --build build
./build/NeuralNet
```

C++17, CMake 3.10+.

## Next

- Backward pass and gradient descent, then training a small classifier end to end
- Loss functions (cross-entropy, MSE)
