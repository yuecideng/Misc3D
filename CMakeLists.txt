cmake_minimum_required(VERSION 3.13)
project(PrimitivesFittingLib VERSION 1.0.0)

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

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find open3d library
set(Open3D_DIR "$ENV{HOME}/Rvbust/Install/Open3D/lib/cmake/Open3D" CACHE FILEPATH "Set to find Open3D")
find_package(Open3D REQUIRED NO_DEFAULT_PATH)

set(CMAKE_INSTALL_PREFIX $ENV{HOME} CACHE PATH "install path")
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})

add_subdirectory(3rdparty)

add_subdirectory(src)

if(BUILD_PYTHON)
    add_subdirectory(python)
endif(BUILD_PYTHON)
