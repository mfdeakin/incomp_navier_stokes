
cmake_minimum_required (VERSION 3.3)
project(energy)

enable_testing()

include_directories(./include)

add_library(ins_solver)

add_executable(fi_test ./src/fi_test.cpp)
set_property(TARGET fi_test PROPERTY CXX_STANDARD 17)
target_link_libraries(fi_test ins_solver gtest pthread)
add_test(flux_int fi_test)

find_package(pybind11 CONFIG REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})
