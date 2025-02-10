cmake_minimum_required(VERSION 3.10)
project(topHat)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Kokkos installation
find_package(Kokkos REQUIRED)

# Add the executable
add_executable(topHat ./src/topHat.cpp)

# Link Kokkos library
target_link_libraries(topHat Kokkos::kokkos)