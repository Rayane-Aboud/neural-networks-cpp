#include "Sho3laDenseLayer.h"
#include <stdexcept>


Sho3laDenseLayer::Sho3laDenseLayer(size_t input_size, size_t output_size)
    :   input_size(input_size), output_size(output_size),
        weights({input_size, output_size}),
        biases({1, output_size})
{
    auto& w = const_cast<std::vector<float>&>(weights.getData());
    for (auto& v : w) v = float(rand())/RAND_MAX - 0.5f;
}


void Sho3laDenseLayer::forward(const Sho3la& input) {
    output = input.dot(weights) + biases;
}