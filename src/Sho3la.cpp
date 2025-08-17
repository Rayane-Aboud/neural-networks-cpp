#include "Sho3la.h"

// ===== Constructors =====
Sho3la::Sho3la(const std::vector<size_t>& shape, float initVal)
    : shape(shape) {
    size_t total = 1;
    for (auto s : shape) total *= s;
    data.assign(total, initVal);
}

// ===== Index helpers =====
size_t Sho3la::flattenIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != shape.size())
        throw std::invalid_argument("Invalid number of indices.");

    size_t idx = 0;
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape[i])
            throw std::out_of_range("Index out of bounds.");
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return idx;
}

std::vector<size_t> Sho3la::computeStrides(const std::vector<size_t>& s) const {
    std::vector<size_t> strides(s.size(), 1);
    for (int i = s.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * s[i + 1];
    return strides;
}

// ===== Access =====
float& Sho3la::at(const std::vector<size_t>& indices) {
    return data[flattenIndex(indices)];
}

const float& Sho3la::at(const std::vector<size_t>& indices) const {
    return data[flattenIndex(indices)];
}

// ===== Broadcasting check =====
bool Sho3la::broadcastable(const Sho3la& other, std::vector<size_t>& outShape) const {
    size_t n = std::max(shape.size(), other.shape.size());
    outShape.resize(n, 1);

    for (size_t i = 0; i < n; i++) {
        size_t dim1 = (i < n - shape.size()) ? 1 : shape[i - (n - shape.size())];
        size_t dim2 = (i < n - other.shape.size()) ? 1 : other.shape[i - (n - other.shape.size())];

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
            return false;

        outShape[i] = std::max(dim1, dim2);
    }
    return true;
}

// ===== Operators =====
Sho3la Sho3la::operator+(const Sho3la& other) const {
    std::vector<size_t> outShape;
    if (!broadcastable(other, outShape))
        throw std::invalid_argument("Shapes not broadcastable for +");

    Sho3la result(outShape);
    auto strides1 = computeStrides(shape);
    auto strides2 = computeStrides(other.shape);
    auto stridesR = computeStrides(outShape);

    for (size_t i = 0; i < result.data.size(); i++) {
        size_t idxR = i;
        std::vector<size_t> coords(outShape.size());
        for (size_t j = 0; j < outShape.size(); j++) {
            coords[j] = idxR / stridesR[j];
            idxR %= stridesR[j];
        }

        std::vector<size_t> idx1(shape.size());
        std::vector<size_t> idx2(other.shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
            size_t offset = outShape.size() - shape.size() + j;
            idx1[j] = (shape[j] == 1) ? 0 : coords[offset];
        }
        for (size_t j = 0; j < other.shape.size(); j++) {
            size_t offset = outShape.size() - other.shape.size() + j;
            idx2[j] = (other.shape[j] == 1) ? 0 : coords[offset];
        }

        result.data[i] = this->at(idx1) + other.at(idx2);
    }
    return result;
}

Sho3la& Sho3la::operator+=(const Sho3la& other) {
    *this = *this + other;
    return *this;
}

Sho3la Sho3la::operator-(const Sho3la& other) const {
    std::vector<size_t> outShape;
    if (!broadcastable(other, outShape))
        throw std::invalid_argument("Shapes not broadcastable for -");

    Sho3la result(outShape);
    auto strides1 = computeStrides(shape);
    auto strides2 = computeStrides(other.shape);
    auto stridesR = computeStrides(outShape);

    for (size_t i = 0; i < result.data.size(); i++) {
        size_t idxR = i;
        std::vector<size_t> coords(outShape.size());
        for (size_t j = 0; j < outShape.size(); j++) {
            coords[j] = idxR / stridesR[j];
            idxR %= stridesR[j];
        }

        std::vector<size_t> idx1(shape.size());
        std::vector<size_t> idx2(other.shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
            size_t offset = outShape.size() - shape.size() + j;
            idx1[j] = (shape[j] == 1) ? 0 : coords[offset];
        }
        for (size_t j = 0; j < other.shape.size(); j++) {
            size_t offset = outShape.size() - other.shape.size() + j;
            idx2[j] = (other.shape[j] == 1) ? 0 : coords[offset];
        }

        result.data[i] = this->at(idx1) - other.at(idx2);
    }
    return result;
}

Sho3la& Sho3la::operator-=(const Sho3la& other) {
    *this = *this - other;
    return *this;
}

// ===== Sum with axis =====
Sho3la Sho3la::sum(size_t axis, bool keepDims) const {
    if (axis >= shape.size())
        throw std::invalid_argument("Invalid axis for sum.");

    std::vector<size_t> newShape = shape;
    newShape[axis] = 1;
    if (!keepDims) newShape.erase(newShape.begin() + axis);

    Sho3la result(newShape, 0.0f);

    auto strides = computeStrides(shape);
    auto stridesR = computeStrides(newShape);

    for (size_t i = 0; i < data.size(); i++) {
        size_t idx = i;
        std::vector<size_t> coords(shape.size());
        for (size_t j = 0; j < shape.size(); j++) {
            coords[j] = idx / strides[j];
            idx %= strides[j];
        }

        std::vector<size_t> reduced(coords);
        reduced[axis] = 0;
        if (!keepDims) reduced.erase(reduced.begin() + axis);

        result.at(reduced) += data[i];
    }

    return result;
}

size_t Sho3la::getSize() const {
    return data.size();
}

// ===== Print =====
void Sho3la::print() const {
    std::cout << "Sho3la(";

    // Print shape
    std::cout << "shape=[";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "], ";

    // Print data nicely depending on dimensions
    std::cout << "data=\n";

    if (shape.size() == 1) {
        // 1D
        std::cout << "[";
        for (size_t i = 0; i < data.size(); i++) {
            std::cout << data[i] << (i + 1 < data.size() ? ", " : "");
        }
        std::cout << "]";
    } 
    else if (shape.size() == 2) {
        // 2D
        size_t rows = shape[0], cols = shape[1];
        for (size_t r = 0; r < rows; r++) {
            std::cout << "[";
            for (size_t c = 0; c < cols; c++) {
                std::cout << data[r * cols + c] << (c + 1 < cols ? ", " : "");
            }
            std::cout << "]\n";
        }
    }
    else if (shape.size() == 3) {
        // 3D (batch-like)
        size_t depth = shape[0], rows = shape[1], cols = shape[2];
        size_t idx = 0;
        for (size_t d = 0; d < depth; d++) {
            std::cout << "Depth " << d << ":\n";
            for (size_t r = 0; r < rows; r++) {
                std::cout << "  [";
                for (size_t c = 0; c < cols; c++) {
                    std::cout << data[idx++] << (c + 1 < cols ? ", " : "");
                }
                std::cout << "]\n";
            }
        }
    }
    else {
        // Fallback for higher dims: flat print
        std::cout << "[";
        for (size_t i = 0; i < data.size(); i++) {
            std::cout << data[i] << (i + 1 < data.size() ? ", " : "");
        }
        std::cout << "]";
    }

    std::cout << ")\n";
}


std::ostream& operator<<(std::ostream& os, const Sho3la& t) {
    os << "Sho3la(shape=[";
    std::vector<size_t> shape = t.getShape();
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i + 1 < shape.size()) os << ", ";
    }
    os << "], data=[";
    const std::vector<float>& data = t.getData();
    for (size_t i = 0; i < data.size(); ++i) {
        os << data[i];
        if (i + 1 < data.size()) os << ", ";
    }
    os << "])";
    return os;
}
Sho3la::Sho3la(const std::vector<size_t>& shape, const std::vector<float>& values)
    : shape(shape), data(values)
{
    size_t total = 1;
    for (auto s : shape) total *= s;
    if (data.size() != total) {
        throw std::invalid_argument("Data size does not match shape");
    }
}


Sho3la Sho3la::dot(const Sho3la& other) const {
    // Case: 2D x 2D
    if (shape.size() == 2 && other.shape.size() == 2) {
        if (shape[1] != other.shape[0])
            throw std::invalid_argument("Matrix dimensions do not align for dot product");

        size_t m = shape[0], n = shape[1], p = other.shape[1];
        Sho3la result({m, p});

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < p; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < n; k++) {
                    sum += this->at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }

    // Case: 3D x 3D (batch matmul)
    if (shape.size() == 3 && other.shape.size() == 3) {
        if (shape[0] != other.shape[0])
            throw std::invalid_argument("Batch sizes must match for 3D dot product");
        if (shape[2] != other.shape[1])
            throw std::invalid_argument("Matrix dimensions do not align for batched dot");

        size_t batch = shape[0], m = shape[1], n = shape[2], p = other.shape[2];
        Sho3la result({batch, m, p});

        for (size_t b = 0; b < batch; b++) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < p; j++) {
                    double sum = 0.0;
                    for (size_t k = 0; k < n; k++) {
                        sum += this->at({b, i, k}) * other.at({b, k, j});
                    }
                    result.at({b, i, j}) = sum;
                }
            }
        }
        return result;
    }

    throw std::invalid_argument("Dot product only supports 2D x 2D or 3D x 3D with equal batch size");
}