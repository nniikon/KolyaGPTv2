#ifndef KGPT_SOFTMAX_H
#define KGPT_SOFTMAX_H

#include "kgpt_matrix.h"
#include "kgpt_operation.h"
#include "kgpt_unary_operation.h"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace kgpt {

template <typename T>
class SoftmaxOperation;

template <typename T>
std::unique_ptr<Operation<BasicMatrixData<T>>> Softmax(GenericMatrix<BasicMatrixData<T>>& input) {
    return std::make_unique<SoftmaxOperation<T>>(input);
}

template <typename T>
class SoftmaxOperation final : public UnaryOperation<BasicMatrixData<T>> {
    using UnaryOperation<BasicMatrixData<T>>::mom_;

public:
    explicit SoftmaxOperation(GenericMatrix<BasicMatrixData<T>>& mom)
        : UnaryOperation<BasicMatrixData<T>>(mom) {}

    const char* name() const override { return "Softmax"; }

    void eval_data(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        const auto& input_data = mom_.data();
        auto& output_data = matrix.data();

        assert(output_data.rows() == input_data.rows() &&
               output_data.cols() == input_data.cols() &&
               "Softmax output dimensions must match input");

        const size_t rows = input_data.rows();
        const size_t cols = input_data.cols();

        for(size_t i = 0; i < rows; ++i) {
            // Numerical stability: subtract row max
            T max_val = *std::max_element(&input_data[i][0], &input_data[i][0]);

            T sum_exp = T(0);
            for(size_t j = 0; j < cols; ++j) {
                sum_exp += std::exp(input_data[i][j] - max_val);
            }

            for(size_t j = 0; j < cols; ++j) {
                output_data[i][j] = std::exp(input_data[i][j] - max_val) / sum_exp;
            }
        }
    }

    void eval_grad(GenericMatrix<BasicMatrixData<T>>& matrix) const override {
        if(!mom_.needs_grad())
            return;

        const auto& output_data = matrix.data();
        const auto& output_grad = matrix.grad();
        auto& input_grad = mom_.grad();

        const size_t rows = output_data.rows();
        const size_t cols = output_data.cols();

        for(size_t i = 0; i < rows; ++i) {
            T sum_term = T(0);
            for(size_t j = 0; j < cols; ++j) {
                sum_term += output_grad[i][j] * output_data[i][j];
            }

            for(size_t j = 0; j < cols; ++j) {
                input_grad[i][j] += output_data[i][j] * (output_grad[i][j] - sum_term);
            }
        }
    }
};

} // namespace kgpt

#endif // KGPT_SOFTMAX_H
