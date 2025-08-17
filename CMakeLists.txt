cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 17)

# Include header files
include_directories(include)

# Source files
file(GLOB SOURCES "src/*.cpp")

add_executable(NeuralNet ${SOURCES})

