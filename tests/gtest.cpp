#include <gtest/gtest.h>
#include "fully_connected_test.h"

TEST(EndToEnd, FullyConnectedSimple) {

    float result = FullyConnectedTest(/*input_size=*/ 10,
                                      /*output_size=*/ 4,
                                      /*n_iterations=*/ 1000,
                                      /*learning_rate=*/ 0.01f,
                                      FullyConnectedType::None,
                                      /*seed=*/ 0);

    ASSERT_LT(result, 0.001f);

    result = FullyConnectedTest(/*input_size=*/ 20,
                                /*output_size=*/ 30,
                                /*n_iterations=*/ 1000,
                                /*learning_rate=*/ 0.01f,
                                FullyConnectedType::None,
                                /*seed=*/ 1);

    ASSERT_LT(result, 0.001f);

    result = FullyConnectedTest(/*input_size=*/ 5,
                                /*output_size=*/ 5,
                                /*n_iterations=*/ 50000,
                                /*learning_rate=*/ 0.01f,
                                FullyConnectedType::None,
                                /*seed=*/ 2);

    ASSERT_LT(result, 0.001f);
}

TEST(EndToEnd, FullyConnectedReLU) {
    float result = FullyConnectedTest(
                                /*input_size=*/ 10,
                                /*output_size=*/ 8,
                                /*n_iterations=*/ 10000,
                                /*learning_rate=*/ 0.2f,
                                FullyConnectedType::Relu,
                                /*seed=*/ 8);

    // Due to "dying ReLU" neurons, loss can't converge to zero.
    ASSERT_LT(result, 0.05f);

    result = FullyConnectedTest(/*input_size=*/ 10,
                                /*output_size=*/ 20,
                                /*n_iterations=*/ 10000,
                                /*learning_rate=*/ 0.2f,
                                FullyConnectedType::Relu,
                                /*seed=*/ 9);

    ASSERT_LT(result, 0.05f);
}

TEST(EndToEnd, FullyConnectedSoftmax) {
    float result = FullyConnectedTest(
                                /*input_size=*/ 10,
                                /*output_size=*/ 4,
                                /*n_iterations=*/ 5000,
                                /*learning_rate=*/ 0.5f,
                                FullyConnectedType::Softmax,
                                /*seed=*/ 15);

    ASSERT_LT(result, 0.001f);

    result = FullyConnectedTest(/*input_size=*/ 20,
                                /*output_size=*/ 30,
                                /*n_iterations=*/ 5000,
                                /*learning_rate=*/ 0.5f,
                                FullyConnectedType::Softmax,
                                /*seed=*/ 20);

    ASSERT_LT(result, 0.001f);
}
