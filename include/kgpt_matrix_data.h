#ifndef KGPT_MATRIX_DATA_H
#define KGPT_MATRIX_DATA_H

#include <vector>
#include <cassert>

namespace kgpt {

template <typename T>
class BasicMatrixData final {
private:
    class RowProxy;
    class ConstRowProxy;

    size_t rows_;
    size_t cols_;
    std::vector<T> data_;

public:
    BasicMatrixData(size_t rows, size_t cols)
        : rows_(rows),
          cols_(cols),
          data_(rows * cols) {
    }

    RowProxy operator[](size_t row) {
        assert(row < rows_ && "Row index out of bounds");
        return RowProxy(*this, row);
    }

    ConstRowProxy operator[](size_t row) const {
        assert(row < rows_ && "Row index out of bounds");
        return ConstRowProxy(*this, row);
    }

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }

    void clear() {
        std::fill(data_.begin(), data_.end(), 0);
    }

private:
    class RowProxy {
        BasicMatrixData& matrix_;
        size_t row_;

    public:
        RowProxy(BasicMatrixData& matrix, size_t row)
            : matrix_(matrix),
              row_(row) {
        }

        T& operator[](size_t col) {
            assert(col < matrix_.cols_ && "Column index out of bounds");
            return matrix_.data_[row_ * matrix_.cols_ + col];
        }
    };

    class ConstRowProxy {
        const BasicMatrixData& matrix_;
        size_t row_;

    public:
        ConstRowProxy(const BasicMatrixData& matrix, size_t row)
            : matrix_(matrix),
              row_(row) {
        }

        const T& operator[](size_t col) const {
            assert(col < matrix_.cols_ && "Column index out of bounds");
            return matrix_.data_[row_ * matrix_.cols_ + col];
        }
    };
};

} // namespace kgpt

#endif // KGPT_MATRIX_DATA_H
