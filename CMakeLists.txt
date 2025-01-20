cmake_minimum_required(VERSION 3.5)

project(sam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_53;code=sm_53)

find_package(OpenCV REQUIRED)

link_directories(C:/Users/william.guillet/Libraries/TensorRT-10.7.0.23/lib)
include_directories(C:/Users/william.guillet/Libraries/TensorRT-10.7.0.23/include)

include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/include/
    ${CUDA_INCLUDE_DIRS}
)

file(GLOB SOURCE
    ./src/*.cpp
)

add_executable(sam ${SOURCE} main.cpp)
target_link_libraries(sam ${OpenCV_LIBS} cudart nvinfer_10 cudart nvonnxparser_10 nvinfer_plugin_10)
