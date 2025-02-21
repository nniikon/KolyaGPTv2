#ifndef KGPT_MATRIX_VECTOR_ADD_H
#define KGPT_MATRIX_VECTOR_ADD_H

#include "kgpt_matrix.h"
#include "kgpt_operation.h"
#include "kgpt_binary_operation.h"
#include <cassert>

namespace kgpt {

template <typename T>
class MatrixVectorAddOperation;

template <typename T>
std::unique_ptr<Operation<BasicMatrixData<T>>> AddVectorToMatrix(
    GenericMatrix<BasicMatrixData<T>>& matrix,
    GenericMatrix<BasicMatrixData<T>>& vector) {
    return std::make_unique<MatrixVectorAddOperation<T>>(matrix, vector);
}

template <typename T>
class MatrixVectorAddOperation final : public BinaryOperation<BasicMatrixData<T>> {

using BinaryOperation<BasicMatrixData<T>>::dad_;
using BinaryOperation<BasicMatrixData<T>>::mom_;

#define matrix_ dad_
#define vector_ mom_

public:
    MatrixVectorAddOperation(GenericMatrix<BasicMatrixData<T>>& dad,
                             GenericMatrix<BasicMatrixData<T>>& mom)
        : BinaryOperation<BasicMatrixData<T>>(dad, mom) {
    }

    const char* name() const override { return "+"; }

    void eval_data(GenericMatrix<BasicMatrixData<T>>& output) const override {
        assert(vector_.data().rows() == 1                     && "Vector must be a row vector");
        assert(matrix_.data().cols() == vector_.data().cols() && "Matrix columns must match vector size");

        matrix_.eval_data();
        vector_.eval_data();

        const size_t rows = matrix_.data().rows();
        const size_t cols = matrix_.data().cols();

        for(size_t i = 0; i < rows; ++i) {
            for(size_t j = 0; j < cols; ++j) {
                output.data()[i][j] = matrix_.data()[i][j] + vector_.data()[0][j];
            }
        }
    }

    void eval_grad(GenericMatrix<BasicMatrixData<T>>& output) const override {
        if (output.needs_grad() == false)
            return;

        const size_t rows = matrix_.data().rows();
        const size_t cols = matrix_.data().cols();

        bool vector_needs_grad = vector_.needs_grad();
        bool matrix_needs_grad = matrix_.needs_grad();

        for(size_t i = 0; i < rows; ++i) {
            for(size_t j = 0; j < cols; ++j) {
                if (vector_needs_grad) matrix_.grad()[i][j] += output.grad()[i][j];
                if (matrix_needs_grad) vector_.grad()[0][j] += output.grad()[i][j];
            }
        }
    }

#undef matrix_
#undef vector_

};

} // namespace kgpt

#endif // KGPT_MATRIX_VECTOR_ADD_H
