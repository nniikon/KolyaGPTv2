#include "../include/kgpt_matrix.h"
#include "../include/kgpt_MSELoss.h"
#include "../include/kgpt_add_vector_to_matrix.h"
#include "../include/kgpt_multiplication.h"
#include "../include/kgpt_matrix_data.h"
#include <cstdio>
#include <gtest/gtest.h>

static void RandomizeMatrix(kgpt::BasicMatrix& mat) {
    size_t N = mat.data().rows();
    size_t M = mat.data().cols();

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            mat.data()[i][j] = static_cast<float>((rand() % 10) - 5) / 5.0f;
        }
    }
}

static size_t GetRandomIntInRange(size_t min, size_t max) {
    return std::rand() % (max - min) + min;
}

static float GetRandomFloatInRange(float min, float max) {
    return std::rand() / static_cast<float>(RAND_MAX) * (max - min) + min;
}

float FullyConnectedTest(size_t input_size, size_t output_size, size_t n_iterations, unsigned int seed) {
    std::srand(seed);

    kgpt::BasicMatrix input          (1,          input_size,  /*needs_grad=*/ false, /*is_trainable=*/ false);
    kgpt::BasicMatrix weights        (input_size, output_size, /*needs_grad=*/ true,  /*is_trainable=*/ true);
    kgpt::BasicMatrix unbiased_output(1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ false);
    kgpt::BasicMatrix biases         (1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ true);
    kgpt::BasicMatrix output         (1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ false);
    kgpt::BasicMatrix ref_output     (1,          output_size, /*needs_grad=*/ false, /*is_trainable=*/ false);
    kgpt::BasicMatrix loss           (1,          1,           /*needs_grad=*/ true,  /*is_trainable=*/ false);

    RandomizeMatrix(input);
    RandomizeMatrix(weights);
    RandomizeMatrix(biases);

    for (size_t i = 0; i < output_size; i++) {
        ref_output.data()[0][i] = GetRandomFloatInRange(-1.0f, 1.0f);
    }

    unbiased_output = kgpt::MatMul           (input, weights);
    output          = kgpt::AddVectorToMatrix(unbiased_output, biases);
    loss            = kgpt::MSELoss          (output, ref_output);

    for (size_t i = 0; i < n_iterations; i++) {
        loss.eval_data_recursive();
        loss.eval_grad_recursive();
        loss.backprop_recursive(0.01f);
    }

    return loss.data()[0][0];
}
