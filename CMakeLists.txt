cmake_minimum_required(VERSION 3.18)
project(rm_pose_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(RM_POSE_SOLVER_FAST_MATH   "Enable fast floating-point optimizations." ON)
option(RM_POSE_SOLVER_LOOP_UNROLL "Enable compiler loop unrolling optimizations." ON)
option(RM_POSE_SOLVER_NATIVE_ARCH "Use native CPU tuning flags (-march=native)." ON)
option(RM_POSE_SOLVER_ENABLE_IPO  "Enable interprocedural optimization/LTO when supported." ON)

# ----- pybind11 -------------------------------------------------------------
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
	set(Python3_FIND_VIRTUALENV FIRST)
	if(DEFINED ENV{VIRTUAL_ENV})
		set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
	elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv")
		set(Python3_ROOT_DIR "${CMAKE_SOURCE_DIR}/.venv")
	endif()

	find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
	execute_process(
		COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
		OUTPUT_VARIABLE _pybind11_cmake_dir
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE _pybind11_cmake_dir_status
	)

	if(NOT _pybind11_cmake_dir_status EQUAL 0 OR _pybind11_cmake_dir STREQUAL "")
		message(FATAL_ERROR "pybind11 not found. Install it in the active Python environment or set pybind11_DIR.")
	endif()

	find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_cmake_dir}")
endif()

# ----- OpenCV ---------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core calib3d)

# ----- Boost (header-only — Math/Tools only, no compiled libs needed) -------
# If Boost is in a non-standard location, pass -DBOOST_ROOT=/path/to/boost
# or set the BOOST_ROOT environment variable before building.
find_package(Boost 1.74 REQUIRED)

# ----- Extension module -----------------------------------------------------
pybind11_add_module(_rm_pose_solver MODULE rm_pose_solver/pose_solver.cpp)

target_include_directories(_rm_pose_solver PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Threads still needed for std::thread used in solve_yaw.
find_package(Threads REQUIRED)
target_link_libraries(_rm_pose_solver PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    # Boost::headers is header-only — no .so/.a to link, just include dirs.
    # Listed explicitly so CMake tracks the dependency correctly.
    Boost::headers
)

if(MSVC)
	target_compile_options(_rm_pose_solver PRIVATE
		/O2 /Ot /GL /fp:fast
	)
	target_link_options(_rm_pose_solver PRIVATE
		/LTCG
	)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	target_compile_options(_rm_pose_solver PRIVATE
		-O3 -fomit-frame-pointer -fstrict-aliasing
	)

	if(RM_POSE_SOLVER_FAST_MATH)
		target_compile_options(_rm_pose_solver PRIVATE
			-ffast-math -fno-math-errno -fno-trapping-math -ffp-contract=fast
		)
	endif()

	if(RM_POSE_SOLVER_LOOP_UNROLL)
		target_compile_options(_rm_pose_solver PRIVATE
			-funroll-loops
		)
	endif()

	if(RM_POSE_SOLVER_NATIVE_ARCH)
		target_compile_options(_rm_pose_solver PRIVATE
			-march=native -mtune=native
		)
	endif()

	target_compile_options(_rm_pose_solver PRIVATE
		-fno-semantic-interposition
	)
endif()

if(RM_POSE_SOLVER_ENABLE_IPO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_output)
	if(_ipo_supported)
		set_property(TARGET _rm_pose_solver PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
	else()
		message(STATUS "IPO/LTO not enabled: ${_ipo_output}")
	endif()
endif()

install(TARGETS _rm_pose_solver DESTINATION rm_pose_solver)
