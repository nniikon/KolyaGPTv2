# KolyaGPTv2

Header-only C++ library mimicking PyTorch with tensor operations and autograd.

## Usage

Simply include the headers directly in your code:

```cpp
// WORK IN PROGRESS
```

## Build & Test

### Using Docker (Recommended)

```sh
podman run --rm -v $(pwd):/app -w /app docker.io/nniikon/kolyagpt:d561fe473123a9290e1a0d44d92de4c630b88f8e \
    sh -c "cmake -B build -S . && cmake --build build --parallel && cd build && ctest --output-on-failure"
```

### Local Build

Ensure you have the required dependencies installed before building:

```sh
cmake -B build -S .
cmake --build build --parallel
cd build && ctest --output-on-failure
```

## Prerequisites

- [CMake](https://cmake.org/download/) **3.28+**
- A **C++20** compatible compiler
- [Google Test](https://github.com/google/googletest) **v1.16.0+**
