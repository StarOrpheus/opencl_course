cmake_minimum_required(VERSION 3.15)
project(opencl_fun)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

add_executable(opencl_fun_a_plus_b main_a_plus_b.c)
target_link_libraries(opencl_fun_a_plus_b OpenCL)
