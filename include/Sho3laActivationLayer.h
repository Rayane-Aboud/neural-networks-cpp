#pragma once
#include "Sho3la.h"


class ISho3laActivationLayer {
protected:
    Sho3la output;

public:
    virtual ~ISho3laActivationLayer() = default;

    virtual void forward(const Sho3la& input);
    const Sho3la& getOutput() const { return output; }
};



class Sho3laReLUActivationLayer: public ISho3laActivationLayer {
private:
    size_t input_size, output_size;
public:
    void forward(const Sho3la& input) override;
};

class Sho3laSoftmaxActivationLayer: public ISho3laActivationLayer {
private:
    size_t input_size, output_size;
    Sho3la output;
public:
    void forward(const Sho3la& input) override;
};