# Sho3la: a neural network library from scratch in C++

I'm writing this small library to understand what PyTorch does under the hood. Everything is written by
hand, without any machine learning framework: the tensors, the layers and the activations. The name
*Sho3la* means "flame" in Algerian Arabic, a small nod to *Torch*.

## What it can do so far

- `Sho3la` is the tensor type. It has a shape with any number of dimensions, element-wise `+` and `-`
  (and `+=`, `-=`), the matrix product `dot`, and `sum` along an axis. A bias can be added to a whole
  batch at once.
- `Sho3laDenseLayer` is a fully connected layer that computes `input · W + b` for a batch.
- ReLU and Softmax are available as activation layers, and they share a common interface.

Here is a small example:

```cpp
Sho3laDenseLayer dense(4, 2);           // 4 inputs, 2 neurons
dense.setWeights(Sho3la({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}));
dense.setBias(Sho3la({1, 2}, {1, 1}));

Sho3la batch({3, 4}, {1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12});
dense.forward(batch);
dense.getOutput().print();              // same result as batch.dot(W) + b
```

## Building it

```bash
cmake -S . -B build
cmake --build build
./build/NeuralNet
```

You need C++17 and CMake 3.10 or newer.

## What's next

I want to add the backward pass and gradient descent next, so I can train a small classifier from start
to end. After that come the loss functions (cross-entropy and mean squared error).
