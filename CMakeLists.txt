cmake_minimum_required(VERSION 3.20)
project(extreem CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

find_package(OpenMP REQUIRED)

add_executable(extreem src/main.cpp)          # add more sources as you go
target_include_directories(extreem PRIVATE include)
target_link_libraries(extreem PRIVATE OpenMP::OpenMP_CXX)

add_executable(run_tests tests/test_main.cpp)
target_include_directories(run_tests PRIVATE include)
target_link_libraries(run_tests PRIVATE OpenMP::OpenMP_CXX)
