#ifndef KGPT_UNARY_OPERATION_H
#define KGPT_UNARY_OPERATION_H

#include "kgpt_operation.h"
#include <iostream>

namespace kgpt {

// eval_data() and eval_grad() must be implemented by derived classes
template <typename MatrixData>
class UnaryOperation : public Operation<MatrixData> {
public:
    virtual const char* name() const = 0;

protected:
    GenericMatrix<MatrixData>& mom_;

public:
    explicit UnaryOperation(GenericMatrix<MatrixData>& mom)
        : mom_(mom) {
    }

    void backprop(GenericMatrix<MatrixData>& matrix, float rate) const override {
        if (matrix.is_trainable()) {
            BasicMatrixBackprop(matrix, rate);
        }
    }

    void backprop_recursive(GenericMatrix<MatrixData>& matrix, float rate) const override {
        backprop(mom_, rate);
        mom_.backprop_recursive(rate);
    }

    void eval_data_recursive(GenericMatrix<MatrixData>& matrix) const override {
        mom_.eval_data_recursive();
        this->eval_data(matrix);
    }

    void eval_grad_recursive(GenericMatrix<MatrixData>& matrix, bool is_root) const override {
        BasicMatrixSetGrad(mom_, 0.0f);

        if (is_root) {
            BasicMatrixSetGrad(matrix, 1.0f);
        }

        this->eval_grad(matrix);

        mom_.eval_grad_recursive(false);
    }

    void dump(GenericMatrix<MatrixData>& matrix, std::ofstream& out) const override {
        BasicMatrixDump(mom_, out);
        mom_.dump(out);

        out << "\top" << this << " [label=\" " << name() << "\"];\n";
        out << "\t" << "Node" << &mom_ << " -> " << "op" << this << ";\n";

        BasicMatrixDump(matrix, out);
        out << "\top" << this << " -> " << "Node" << &matrix << ";\n";
    }

    // dad() returns nullptr for unary operations
    const GenericMatrix<MatrixData>* dad() const override { return nullptr; };
    const GenericMatrix<MatrixData>* mom() const override { return &mom_; };
};

} // namespace kgpt

#endif // KGPT_UNARY_OPERATION_H
