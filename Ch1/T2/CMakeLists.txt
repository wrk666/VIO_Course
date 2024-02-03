cmake_minimum_required(VERSION 3.17)
project(T2)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

# Eigen
include_directories("/usr/include/eigen3")

add_executable(main main.cpp)
target_link_libraries(main
        Sophus::Sophus)
