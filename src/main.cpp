#include <iostream>
#include "Sho3la.h"
#include "Sho3laDenseLayer.h"

int main() {
    try {
        // Example: batch = 3, input_size = 4, output_size = 2
        Sho3laDenseLayer dense(4, 2);

        // Initialize weights and bias (just random/simple values for test)
        /*
        two neurons
        4 weights
        */
        Sho3la weights({4, 2}, {1, 2,
                                3, 4,
                                5, 6,
                                7, 8});
        Sho3la bias({1, 2}, {1, 1});
        dense.setWeights(weights);
        dense.setBias(bias);

        // Create input: (3,4)
        /*
        3 batches of data
        4 features
        */
        Sho3la input({3, 4}, {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,10,11,12
        });

        // Forward pass
        dense.forward(input);

        // Get output
        Sho3la output = dense.getOutput();
        output.print();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
