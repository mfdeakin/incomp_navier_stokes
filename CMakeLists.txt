
cmake_minimum_required (VERSION 3.3)
project(energy)

enable_testing()

include_directories(./include)

add_executable(fi_test ./src/fi_test.cpp)
target_link_libraries(fi_test gtest pthread)
add_test(flux_int fi_test)
set_property(TARGET fi_test PROPERTY CXX_STANDARD 17)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(ins_solver ./src/ins_solver)
set_property(TARGET ins_solver PROPERTY CXX_STANDARD 17)

configure_file(./scripts/part1.py part1.py COPYONLY)
