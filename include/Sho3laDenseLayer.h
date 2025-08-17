#include <cstddef> 
#include "Sho3la.h"


class Sho3laDenseLayer {
private:
    size_t  input_size, // the number of features
            output_size;// is the number of neurons
    Sho3la weights, biases, output;  // matrix of size (input_size x output_size)

public:
    Sho3laDenseLayer(size_t input_size, size_t output_size);
    void forward(const Sho3la& input);

    void setWeights(const Sho3la& w) {
        if (w.getShape() != std::vector<size_t>{input_size, output_size})
            throw std::invalid_argument("Weights shape mismatch");
        weights = w;
    }

    // Setter for biases
    void setBias(const Sho3la& b) {
        if (b.getShape() != std::vector<size_t>{1, output_size})
            throw std::invalid_argument("Bias shape mismatch");
        biases = b;
    }

    // Getter for output
    const Sho3la& getOutput() const { return output; }

};
