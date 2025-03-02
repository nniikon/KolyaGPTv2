# KolyaGPTv2

Header-only C++ library mimicking PyTorch with tensor operations and autograd.

## Usage

Simply include the headers directly in your code:

```cpp
// WORK IN PROGRESS
```

## Build & Test

### Prerequisites

- CMake 3.28+

- C++20 compiler

- Google Test

### Docker (recommended)

```sh
podman run --rm -v $(pwd):/app -w /app nniikon/kolyagpt:latest sh -c "cmake -B build -S . && cmake --build build --parallel && cd build && ctest --output-on-failure"
```

### Local Build
```sh
cmake -B build -S .
cmake --build build --parallel
cd build && ctest --output-on-failure
```
