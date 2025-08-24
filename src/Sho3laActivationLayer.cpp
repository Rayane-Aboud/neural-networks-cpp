#include "Sho3laActivationLayer.h"



void Sho3laReLUActivationLayer::forward(const Sho3la& input) {
    const auto& inputData = input.getData();
    Sho3la result(input.getShape());    
    for (size_t i = 0; i < inputData.size(); i++){
        result.at({i}) = std::max(0.0f, inputData[i]);
    }
    output = result;
}

