#ifndef KGPT_OPERATION_H
#define KGPT_OPERATION_H

#include "kgpt_matrix_data.h"
#include <fstream>

namespace kgpt {

template <typename MatrixData>
class GenericMatrix;

template <typename MatrixData>
class Operation {
public:
    virtual ~Operation() = default;

    virtual void eval_data(GenericMatrix<MatrixData>& matrix) const = 0;
    virtual void eval_grad(GenericMatrix<MatrixData>& matrix) const = 0;
    virtual void backprop (GenericMatrix<MatrixData>& matrix, float rate) const = 0;

    virtual void eval_data_recursive  (GenericMatrix<MatrixData>& matrix) const = 0;
    virtual void eval_grad_recursive  (GenericMatrix<MatrixData>& matrix, bool is_root = true) const = 0;
    virtual void backprop_recursive   (GenericMatrix<MatrixData>& matrix, float rate) const = 0;

    virtual void dump(GenericMatrix<MatrixData>& matrix, std::ofstream& out) const = 0;

    // Note: may return nullptr
    virtual const GenericMatrix<MatrixData>* dad() const = 0;
    virtual const GenericMatrix<MatrixData>* mom() const = 0;
};

template <typename T>
void BasicMatrixBackprop(GenericMatrix<BasicMatrixData<T>>& matrix, float rate) {
    for (size_t i = 0; i < matrix.data().rows(); i++) {
        for (size_t j = 0; j < matrix.data().cols(); j++) {
            matrix.data()[i][j] -= rate * matrix.grad()[i][j];
        }
    }
}

template <typename T>
void BasicMatrixDump(GenericMatrix<BasicMatrixData<T>>& matrix, std::ofstream& out) {
    const size_t rows = matrix.data().rows();
    const size_t cols = matrix.data().cols();

    // Lambda that prints one group (data or grads)
    auto printMatrixGroup = [&, rows, cols](const std::string &name, const auto &group) {
        out << name << ":|";
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                out << group[i][j];
                if (j < cols - 1)
                    out << ", ";
            }
            if (i < rows - 1)
                out << "|";
        }
    };

    out << "Node" << &matrix << " [label=\"{";
    printMatrixGroup("Data", matrix.data());
    out << "} | {";
    printMatrixGroup("Grads", matrix.grad());
    out << "}\"];\n";
}

template <typename T>
void BasicMatrixSetGrad(GenericMatrix<BasicMatrixData<T>>& matrix, T value) {
    size_t rows = matrix.data().rows();
    size_t cols = matrix.data().cols();

    for (size_t i = 0; i < rows; i++) {    
        for (size_t j = 0; j < cols; j++) {
            matrix.grad()[i][j] = value;
        }
    }
}

} // namespace

#endif // KGPT_OPERATION_H
