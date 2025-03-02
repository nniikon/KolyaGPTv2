#include "fully_connected_test.h"
#include "../include/kgpt_matrix.h"
#include "../include/kgpt_MSELoss.h"
#include "../include/kgpt_add_vector_to_matrix.h"
#include "../include/kgpt_multiplication.h"
#include "../include/kgpt_matrix_data.h"
#include "../include/kgpt_ReLU.h"
#include "../include/kgpt_softmax.h"
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

float FullyConnectedTest(size_t input_size,
                         size_t output_size,
                         size_t n_iterations,
                         float learning_rate,
                         FullyConnectedType type,
                         unsigned int seed) {
    std::srand(seed);

    kgpt::BasicMatrix input          (1,          input_size,  /*needs_grad=*/ false, /*is_trainable=*/ false);
    kgpt::BasicMatrix weights        (input_size, output_size, /*needs_grad=*/ true,  /*is_trainable=*/ true);
    kgpt::BasicMatrix unbiased_output(1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ false);
    kgpt::BasicMatrix biases         (1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ true);
    kgpt::BasicMatrix output         (1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ false);
    kgpt::BasicMatrix norm_output    (1,          output_size, /*needs_grad=*/ true,  /*is_trainable=*/ true);
    kgpt::BasicMatrix ref_output     (1,          output_size, /*needs_grad=*/ false, /*is_trainable=*/ false);
    kgpt::BasicMatrix loss           (1,          1,           /*needs_grad=*/ true,  /*is_trainable=*/ false);

    RandomizeMatrix(input);
    RandomizeMatrix(weights);
    RandomizeMatrix(biases);

    std::vector<float> temp(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        temp[i] = GetRandomFloatInRange(0.0f, 1.0f);
    }

    float sum = 0.0f;
    for (float val : temp) sum += val;

    assert(sum > 0.01f);

    for (size_t i = 0; i < output_size; ++i) {
        ref_output.data()[0][i] = temp[i] / sum;
    }

    unbiased_output = kgpt::MatMul           (input, weights);
    output          = kgpt::AddVectorToMatrix(unbiased_output, biases);

    switch (type) {
        case FullyConnectedType::None:
            loss = kgpt::MSELoss(output, ref_output);
            break;
        case FullyConnectedType::Relu:
            norm_output = kgpt::ReLU(output);
            loss = kgpt::MSELoss(norm_output, ref_output);
            break;
        case FullyConnectedType::Softmax:
            norm_output = kgpt::Softmax(output);
            loss = kgpt::MSELoss(norm_output, ref_output);
            break;
    }

    for (size_t i = 0; i < n_iterations; i++) {
        loss.eval_data_recursive();
        loss.eval_grad_recursive();
        loss.backprop_recursive(learning_rate);
    }

    fprintf(stderr, "\t" "output | expected\n");
    for (size_t i = 0; i < output_size; i++) {
        fprintf(stderr, "\t" "%5.2f | %5.2f\n", norm_output.data()[0][i], ref_output.data()[0][i]);
    }

    return loss.data()[0][0];
}
