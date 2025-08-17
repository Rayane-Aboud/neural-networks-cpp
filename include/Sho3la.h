#ifndef SHO3LA_H
#define SHO3LA_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>

class Sho3la {
private:
    std::vector<float> data;
    std::vector<size_t> shape;

    size_t flattenIndex(const std::vector<size_t>& indices) const;
    std::vector<size_t> computeStrides(const std::vector<size_t>& s) const;

    // New helper
    bool broadcastable(const Sho3la& other, std::vector<size_t>& outShape) const;

public:
    Sho3la() = default;
    Sho3la(const std::vector<size_t>& shape, float initVal = 0.0f);
    Sho3la(const std::vector<size_t>& shape, const std::vector<float>& values);

    float& at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    const std::vector<size_t>& getShape() const { return shape; }
    size_t getSize() const;
    const std::vector<float>& getData() const { return data; }


    // Operations
    Sho3la operator+(const Sho3la& other) const;
    Sho3la& operator+=(const Sho3la& other);
    Sho3la operator-(const Sho3la& other) const;
    Sho3la& operator-=(const Sho3la& other);

    //
    Sho3la dot(const Sho3la& other) const;

    Sho3la sum(size_t axis, bool keepDims = false) const;

    void print() const;
};

std::ostream& operator<<(std::ostream& os, const Sho3la& t);


#endif
