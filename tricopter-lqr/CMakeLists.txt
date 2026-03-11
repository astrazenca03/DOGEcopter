cmake_minimum_required(VERSION 3.16)
project(tricopter_lqr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -O2)

# --- Eigen3 via FetchContent ---
include(FetchContent)

find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally — fetching via FetchContent")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# --- yaml-cpp via FetchContent ---
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found locally — fetching via FetchContent")
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG        0.8.0
        GIT_SHALLOW    TRUE
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# --- Library with all source files ---
set(LIB_SOURCES
    src/config.cpp
    src/control_allocation.cpp
    src/trim.cpp
    src/dynamics.cpp
    src/integrator.cpp
    src/lqr.cpp
    src/pid.cpp
    src/simulation.cpp
)

add_library(tricopter_lib STATIC ${LIB_SOURCES})
target_include_directories(tricopter_lib PUBLIC include)
target_link_libraries(tricopter_lib PUBLIC Eigen3::Eigen yaml-cpp::yaml-cpp)

# --- Main executable ---
add_executable(tricopter_sim src/main.cpp)
target_link_libraries(tricopter_sim PRIVATE tricopter_lib)

# --- Unit tests ---
add_executable(test_main tests/test_main.cpp)
target_link_libraries(test_main PRIVATE tricopter_lib)
