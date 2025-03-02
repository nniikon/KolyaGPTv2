#ifndef KGPT_MATRIX_H
#define KGPT_MATRIX_H

#include "kgpt_operation.h"
#include <iostream>
#include <memory>
#include <fstream>

namespace kgpt {

using BasicMatrix = GenericMatrix<BasicMatrixData<float>>;

template <typename MatrixData>
class GenericMatrix final {
private:
    MatrixData data_;
    MatrixData grad_;
    std::unique_ptr<Operation<MatrixData>> operation_ = nullptr;
    bool needs_grad_   = true;
    bool is_trainable_ = true;

public:
    GenericMatrix(size_t rows, size_t cols, bool needs_grad = true, bool is_trainable = true)
        : data_(rows, cols),
          grad_(rows, cols),
          needs_grad_(needs_grad),
          is_trainable_(is_trainable) {
        reset();
    }

    MatrixData&       data()       { return data_; }
    const MatrixData& data() const { return data_; }

    MatrixData&       grad()       { return grad_; }
    const MatrixData& grad() const { return grad_; }

    bool needs_grad()   const { return needs_grad_; }
    bool is_trainable() const { return is_trainable_; }

    void eval_data()          { if(operation_)                  operation_->eval_data(*this); }
    void eval_grad()          { if(operation_ && needs_grad_  ) operation_->eval_grad(*this); }
    void backprop(float rate) { if(operation_ && is_trainable_) operation_->backprop (*this, rate); }

    void eval_data_recursive()                    { if(operation_) operation_->eval_data_recursive(*this); }
    void eval_grad_recursive(bool is_root = true) { if(operation_) operation_->eval_grad_recursive(*this, is_root); }
    void backprop_recursive(float rate)           { if(operation_) operation_->backprop_recursive (*this, rate); }

    void dump(std::ofstream& out) {
        if (operation_) operation_->dump(*this, out);
    }

    void dump_recursive(const std::string& filename) {
        std::ofstream out(filename + ".dot");

        out << "digraph G {\n";
        out << "\tnode [shape=record];\n";

        dump(out);

        out << "}\n";
        out.close();

        std::string compile_cmd = "dot -Tpng " + filename + ".dot" + " -o " + filename + ".png";
        system(compile_cmd.c_str());
    }

    void reset() {
        data_.clear();
        grad_.clear();
    }

    GenericMatrix& operator=(std::unique_ptr<Operation<MatrixData>> operation) {
        GenericMatrix result(data().rows(), data().cols());

        operation_ = std::move(operation);

        return *this;
    }
};


} // namespace

#endif // KGPT_MATRIX_H
