cmake_minimum_required(VERSION 3.10)
project(Pudu3D VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "cmake build type: " ${CMAKE_BUILD_TYPE})

if(UNIX)
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/lib)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

option(BUILD_PYTHON "Build Python bindings" ON)
option(ENABLE_LOG_PRINT "Print log on terminal" ON)

###############################################################################
# Find 3rdparty dependencies

# Find OpenMP for parallel computing
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Find open3d 
find_package(Open3D REQUIRED)

# Find pybind11 for Python wrappers
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

set(CMAKE_INSTALL_PREFIX $ENV{HOME} CACHE PATH "install path")
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})

# enable log printing on terminal
if(ENABLE_LOG_PRINT)
    message(STATUS "Enable log printed in terminal")
    add_definitions(-DENABLE_LOGGER_PRINT)
endif(ENABLE_LOG_PRINT)

add_subdirectory(3rdparty)

include_directories(${SPDLOG_INCLUDE_DIRS})
add_subdirectory(src)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif(BUILD_PYTHON)
