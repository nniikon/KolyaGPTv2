#ifndef KGPT_MATRIX_MULTIPLICATION_H
#define KGPT_MATRIX_MULTIPLICATION_H

#include "kgpt_matrix.h"
#include "kgpt_operation.h"
#include "kgpt_binary_operation.h"
#include <cassert>

namespace kgpt {

template <typename T>
class MatrixMultiplicationOperation;

template <typename T>
std::unique_ptr<Operation<BasicMatrixData<T>>> MatMul(GenericMatrix<BasicMatrixData<T>>& dad,
                                                    GenericMatrix<BasicMatrixData<T>>& mom) {
    return std::make_unique<MatrixMultiplicationOperation<T>>(dad, mom);
}

template <typename T>
class MatrixMultiplicationOperation final : public BinaryOperation<BasicMatrixData<T>> {

using BinaryOperation<BasicMatrixData<T>>::dad_;
using BinaryOperation<BasicMatrixData<T>>::mom_;

public:
    MatrixMultiplicationOperation(GenericMatrix<BasicMatrixData<T>>& dad,
                                  GenericMatrix<BasicMatrixData<T>>& mom)
        : BinaryOperation<BasicMatrixData<T>>(dad, mom) {
    }

    const char* name() const override { return "*"; }

    void eval_data(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        const size_t dad_rows   = dad_.data().rows();
        const size_t shared_dim = dad_.data().cols();
        const size_t mom_cols   = mom_.data().cols();

        assert(shared_dim == mom_.data().rows() && "Matrix multiplication dimension mismatch");
        assert(matrix.data().cols() == mom_cols && "Matrix multiplication dimension mismatch");
        assert(matrix.data().rows() == dad_rows && "Matrix multiplication dimension mismatch");

        for(size_t i = 0; i < dad_rows; i++) {
            for(size_t j = 0; j < mom_cols; j++) {
                matrix.data()[i][j] = T(0);
            }
        }

        for(size_t i = 0; i < dad_rows; i++) {
            for(size_t k = 0; k < shared_dim; k++) {
                const T dad_val = dad_.data()[i][k];
                for(size_t j = 0; j < mom_cols; j++) {
                    matrix.data()[i][j] += dad_val * mom_.data()[k][j];
                }
            }
        }
    }

    void eval_grad(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        if (matrix.needs_grad() == false)
            return;

        const size_t dad_rows   = dad_.data().rows();
        const size_t shared_dim = dad_.data().cols();
        const size_t mom_cols   = mom_.data().cols();

        // Gradient for dad: (grad * mom^T)
        if (dad_.needs_grad()) {
            for(size_t i = 0; i < dad_rows; i++) {
                for(size_t j = 0; j < mom_cols; j++) {
                    const T grad_val = matrix.grad()[i][j];
                    for(size_t k = 0; k < shared_dim; k++) {
                        dad_.grad()[i][k] += grad_val * mom_.data()[k][j];
                    }
                }
            }
        }

        // Gradient for mom: (dad^T * grad)
        if (mom_.needs_grad()) {
            for(size_t k = 0; k < shared_dim; k++) {
                for(size_t i = 0; i < dad_rows; i++) {
                    const T dad_val = dad_.data()[i][k];
                    for(size_t j = 0; j < mom_cols; j++) {
                        mom_.grad()[k][j] += dad_val * matrix.grad()[i][j];
                    }
                }
            }
        }
    }
};

} // namespace kgpt

#endif // KGPT_MATRIX_MULTIPLICATION_H
