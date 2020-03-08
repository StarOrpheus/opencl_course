cmake_minimum_required(VERSION 3.15)
project(opencl_fun)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")

add_executable(opencl_fun_a_plus_b main_a_plus_b.c)
target_link_libraries(opencl_fun_a_plus_b OpenCL)

add_executable(opencl_fun_array_sum main_array_sum.c)
target_link_libraries(opencl_fun_array_sum OpenCL)

add_executable(opencl_fun_gemm1 gemm1.c)
target_link_libraries(opencl_fun_gemm1 OpenCL)

add_executable(opencl_fun_gemm2 gemm2.c)
target_link_libraries(opencl_fun_gemm2 OpenCL)

add_executable(opencl_fun_gemm3 gemm3.c)
target_link_libraries(opencl_fun_gemm3 OpenCL)