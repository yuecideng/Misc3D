cmake_minimum_required(VERSION 3.10)
project(Pudu3D VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

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

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(ENABLE_LOG_FILE "Save log file into log/misc3d.txt" OFF)

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
if(ENABLE_LOG_FILE)
    message(STATUS "Enable log file saving")
    add_definitions(-DENABLE_LOG_FILE)
endif(ENABLE_LOG_FILE)

add_subdirectory(3rdparty)

include_directories(${SPDLOG_INCLUDE_DIRS})
add_subdirectory(src)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif(BUILD_PYTHON)

# install python examples 
install(DIRECTORY examples
    DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/"
    USE_SOURCE_PERMISSIONS
    )

