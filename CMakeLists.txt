
cmake_minimum_required (VERSION 3.3)
project(energy)

enable_testing()

include_directories(./include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

add_executable(explicit ./src/explicit.cpp)

add_executable(energy_tester ./src/test.cpp)
add_test(energy_test energy_tester)

target_link_libraries(energy_tester gtest pthread)
target_link_libraries(explicit pybind11::embed)
