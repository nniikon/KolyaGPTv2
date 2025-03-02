FROM ubuntu:24.04

RUN apt update && \
    apt install -y \
        build-essential \
        unzip \
        wget \
        cmake && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip -O gtest.zip && \
    unzip gtest.zip && \
    cd googletest-03597a01ee50ed33e9dfd640b249b4be3799d395 && \
    cmake -B build -S . && \
    cmake --build build --target install && \
    cd .. && rm -rf googletest-* gtest.zip

WORKDIR /workspace
