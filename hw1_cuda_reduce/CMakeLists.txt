# 最低要求 CMake 3.16
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_COMPILER hipcc CACHE STRING "HIP compiler")
project(hw1_hip_reduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_reduce
    test_reduce.cpp
    reduce.cpp
)
