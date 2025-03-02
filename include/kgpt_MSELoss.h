#ifndef KGPT_MSELOSS_H
#define KGPT_MSELOSS_H

#include "kgpt_matrix.h"
#include "kgpt_operation.h"
#include "kgpt_binary_operation.h"
#include <cassert>

namespace kgpt {

template <typename T>
class MSELossOperation;

template <typename T>
std::unique_ptr<Operation<BasicMatrixData<T>>> MSELoss(GenericMatrix<BasicMatrixData<T>>& dad,
                                                       GenericMatrix<BasicMatrixData<T>>& mom) {
    return std::make_unique<MSELossOperation<T>>(dad, mom);
}

template <typename T>
class MSELossOperation final : public BinaryOperation<BasicMatrixData<T>> {

using BinaryOperation<BasicMatrixData<T>>::dad_;
using BinaryOperation<BasicMatrixData<T>>::mom_;

private:
    T n_elements_ = T(0);

public:
    MSELossOperation(GenericMatrix<BasicMatrixData<T>>& dad,
                     GenericMatrix<BasicMatrixData<T>>& mom)
        : BinaryOperation<BasicMatrixData<T>>(dad, mom),
          n_elements_(static_cast<T>(dad.data().rows() * dad.data().cols())) {
    }

    const char* name() const override { return "MSELoss"; }

    void eval_data(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        assert(dad_.data().rows() == mom_.data().rows() &&
               dad_.data().cols() == mom_.data().cols() &&
               "Input dimensions must match for MSELoss");
        assert(matrix.data().rows() == 1 && matrix.data().cols() == 1 &&
               "MSELoss output must be 1x1 matrix");

        const size_t rows = dad_.data().rows();
        const size_t cols = dad_.data().cols();
        T sum = T(0);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T diff = dad_.data()[i][j] - mom_.data()[i][j];
                sum += diff * diff;
            }
        }

        matrix.data()[0][0] = sum / n_elements_;
    }

    void eval_grad(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        const T grad = matrix.grad()[0][0];
        const T scale = T(2) * grad / n_elements_;

        const size_t rows = dad_.data().rows();
        const size_t cols = dad_.data().cols();

        bool dad_needs_grad = dad_.needs_grad();
        bool mom_needs_grad = mom_.needs_grad();

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const T diff = dad_.data()[i][j] - mom_.data()[i][j];
                if (dad_needs_grad) dad_.grad()[i][j] += scale * diff;
                if (mom_needs_grad) mom_.grad()[i][j] -= scale * diff;
            }
        }
    }
};

} // namespace kgpt

#endif // KGPT_MSELOSS_H
