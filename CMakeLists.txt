
cmake_minimum_required (VERSION 3.3)
project(energy)

enable_testing()

include_directories(./include)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(ins_solver ./src/ins_solver)
set_property(TARGET ins_solver PROPERTY CXX_STANDARD 17)

add_executable(plot_ts ./src/plot_ts.cpp)
set_property(TARGET plot_ts PROPERTY CXX_STANDARD 17)
target_link_libraries(plot_ts pybind11::embed)

configure_file(./scripts/part1.py part1.py COPYONLY)
