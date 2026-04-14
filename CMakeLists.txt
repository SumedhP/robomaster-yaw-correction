cmake_minimum_required(VERSION 3.18)
project(rm_pose_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----- pybind11 (fetched by scikit-build-core via pyproject.toml) ----------
find_package(pybind11 CONFIG REQUIRED)

# ----- OpenCV ---------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)

# ----- Extension module -----------------------------------------------------
pybind11_add_module(_rm_pose_solver MODULE pose_solver.cpp)

target_include_directories(_rm_pose_solver PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(_rm_pose_solver PRIVATE ${OpenCV_LIBS})

# Place the .so next to the Python package so the import works
install(TARGETS _rm_pose_solver DESTINATION rm_pose_solver)
