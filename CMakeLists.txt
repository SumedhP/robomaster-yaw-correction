cmake_minimum_required(VERSION 3.18)
project(rm_pose_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# pybind11 — try system install first, fall back to pip-installed copy
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pb11_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED HINTS "${_pb11_dir}")
endif()

find_package(Boost  1.74 REQUIRED)          # header-only (Math/Tools)
find_package(Eigen3 3.3  REQUIRED NO_MODULE)
find_package(Threads     REQUIRED)

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------

pybind11_add_module(_rm_pose_solver MODULE rm_pose_solver/pose_solver.cpp)

target_link_libraries(_rm_pose_solver PRIVATE
    Threads::Threads
    Boost::headers
    Eigen3::Eigen)

target_compile_definitions(_rm_pose_solver PRIVATE
    EIGEN_NO_DEBUG
    EIGEN_UNROLLING_LIMIT=100)

target_compile_options(_rm_pose_solver PRIVATE
    -O3
    -march=native
    -mtune=native
    -ffast-math
    -fno-math-errno
    -fno-trapping-math
    -ffp-contract=fast
    -fomit-frame-pointer
    -fstrict-aliasing
    -fno-semantic-interposition
    -funroll-loops)

# LTO — worth it on Jetson for the pybind11 glue overhead
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
if(_ipo_ok)
    set_property(TARGET _rm_pose_solver PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "LTO not available: ${_ipo_msg}")
endif()

install(TARGETS _rm_pose_solver DESTINATION rm_pose_solver)