cmake_minimum_required(VERSION 3.3)

project(OpenEarthBenchmarks VERSION 1.0
  DESCRIPTION "Open Earth Benchmarks"
  LANGUAGES CXX CUDA)

#find_package(OpenMP)
find_package(CUDA REQUIRED)

include_directories(include)
include_directories(kernels)

add_compile_options(-O3)
function(add_example example_name)
  add_executable(${example_name} src/${example_name}.cu)
  # target_link_libraries(${example_name})
  # target_link_libraries(${example_name} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

#set(CMAKE_CUDA_FLAGS " -G -g -line-info ")
# add_example(laplace)
add_example(hdiffsa)
add_example(hdiffsmag)
add_example(hadvuv)
add_example(hadvuv5th)
add_example(fastwavesuv)
# add_example(uvbke)
# add_example(p_grad_c)
# add_example(nh_p_grad)
# add_example(fvtp2d)
