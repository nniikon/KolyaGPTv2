#ifndef KGPT_SUBTRACTION_H
#define KGPT_SUBTRACTION_H

#include "kgpt_matrix.h"
#include "kgpt_operation.h"
#include "kgpt_binary_operation.h"
#include <cassert>

namespace kgpt {

template <typename T>
class SubtractionOperation;

template <typename T>
std::unique_ptr<Operation<BasicMatrixData<T>>> Sub(GenericMatrix<BasicMatrixData<T>>& dad,
                                                   GenericMatrix<BasicMatrixData<T>>& mom) {
    return std::make_unique<SubtractionOperation<T>>(dad, mom);
}

template <typename T>
class SubtractionOperation final : public BinaryOperation<BasicMatrixData<T>> {

using BinaryOperation<BasicMatrixData<T>>::dad_;
using BinaryOperation<BasicMatrixData<T>>::mom_;

public:
    SubtractionOperation(GenericMatrix<BasicMatrixData<T>>& dad,
                         GenericMatrix<BasicMatrixData<T>>& mom)
        : BinaryOperation<BasicMatrixData<T>>(dad, mom) {
    }

    const char* name() const override { return "-"; }

    void eval_data(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        assert(dad_.data().cols() == mom_.data().cols() && "Column count mismatch");
        assert(dad_.data().rows() == mom_.data().rows() && "Row count mismatch");

        size_t rows = mom_.data().rows();
        size_t cols = mom_.data().cols();

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                matrix.data()[i][j] = dad_.data()[i][j] - mom_.data()[i][j];
            }
        }
    }

    void eval_grad(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        if (matrix.needs_grad() == false)
            return;

        assert(dad_.data().cols() == mom_.data().cols() && "Column count mismatch");
        assert(dad_.data().rows() == mom_.data().rows() && "Row count mismatch");

        size_t rows = mom_.data().rows();
        size_t cols = mom_.data().cols();

        bool dad_needs_grad = dad_.needs_grad();
        bool mom_needs_grad = mom_.needs_grad();

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                if (dad_needs_grad) dad_.grad()[i][j] += matrix.grad()[i][j];
                if (mom_needs_grad) mom_.grad()[i][j] -= matrix.grad()[i][j];
            }
        }
    }
};

} // namespace kgpt

#endif // KGPT_SUBTRACTION_H
