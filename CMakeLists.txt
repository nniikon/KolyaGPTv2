cmake_minimum_required(VERSION 3.30)
project(KolyaGPT)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
