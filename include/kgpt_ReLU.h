#ifndef KGPT_RELU_H
#define KGPT_RELU_H

#include "kgpt_matrix.h"
#include "kgpt_operation.h"
#include "kgpt_unary_operation.h"
#include <cassert>
#include <algorithm>  // For std::max

namespace kgpt {

template <typename T>
class ReLUOperation;

template <typename T>
std::unique_ptr<Operation<BasicMatrixData<T>>> ReLU(GenericMatrix<BasicMatrixData<T>>& input) {
    return std::make_unique<ReLUOperation<T>>(input);
}

template <typename T>
class ReLUOperation final : public UnaryOperation<BasicMatrixData<T>> {

using UnaryOperation<BasicMatrixData<T>>::mom_;

public:
    explicit ReLUOperation(GenericMatrix<BasicMatrixData<T>>& input)
        : UnaryOperation<BasicMatrixData<T>>(input) {
    }

    const char* name() const override { return "ReLU"; }

    void eval_data(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        const auto& input_data = mom_.data();
        auto& output_data = matrix.data();

        assert(output_data.rows() == input_data.rows() &&
               output_data.cols() == input_data.cols() &&
               "ReLU output dimensions must match input");

        const size_t rows = input_data.rows();
        const size_t cols = input_data.cols();

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                output_data[i][j] = std::max(T(0), input_data[i][j]);
            }
        }
    }

    void eval_grad(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        if (!mom_.needs_grad())
            return;

        const auto& input_data = mom_.data();
        const auto& output_grad = matrix.grad();
        auto& input_grad = mom_.grad();

        const size_t rows = input_data.rows();
        const size_t cols = input_data.cols();

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if (input_data[i][j] > T(0)) {
                    input_grad[i][j] += output_grad[i][j];
                }
            }
        }
    }
};

} // namespace kgpt

#endif // KGPT_RELU_H
