#include <gtest/gtest.h>
#include "fully_connected_test.h"

TEST(EndToEnd, FullyConnected) {

    float result = FullyConnectedTest(/*input_size=*/ 10,
                                      /*output_size=*/ 4,
                                      /*n_iterations=*/ 1000,
                                      /*seed=*/ 0);
    
    ASSERT_LT(result, 0.001f);

    result = FullyConnectedTest(/*input_size=*/ 20,
                                /*output_size=*/ 30,
                                /*n_iterations=*/ 1000,
                                /*seed=*/ 1);
    
    ASSERT_LT(result, 0.001f);

    result = FullyConnectedTest(/*input_size=*/ 5,
                                /*output_size=*/ 5,
                                /*n_iterations=*/ 50000,
                                /*seed=*/ 2);
    
    ASSERT_LT(result, 0.001f);
}
