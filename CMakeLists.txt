cmake_minimum_required(VERSION 3.28)
project(KolyaGPT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tests
    ${PROJECT_SOURCE_DIR}/tests/gtest.cpp
    ${PROJECT_SOURCE_DIR}/tests/fully_connected_test.cpp
)

target_include_directories(tests PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(tests
    GTest::gtest
    GTest::gtest_main
)

enable_testing()
gtest_discover_tests(tests)
