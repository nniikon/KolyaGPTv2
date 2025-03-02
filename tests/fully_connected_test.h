#ifndef KGPT_FULLY_CONNECTED_TEST_H
#define KGPT_FULLY_CONNECTED_TEST_H

#include <cstddef>

enum class FullyConnectedType {
    None,
    Relu,
    Softmax,
};

float FullyConnectedTest(size_t input_size,
                         size_t output_size,
                         size_t n_iterations,
                         float learning_rate,
                         FullyConnectedType type,
                         unsigned int seed);

#endif // KGPT_FULLY_CONNECTED_TEST_H
