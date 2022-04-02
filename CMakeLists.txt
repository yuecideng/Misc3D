cmake_minimum_required(VERSION 3.10)
project(Misc3D VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "cmake build type: " ${CMAKE_BUILD_TYPE})

message(STATUS "CPU architecture: " ${CMAKE_SYSTEM_PROCESSOR})
if(NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
    add_definitions(-DARM_CPU)
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
if(UNIX)
    set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${PROJECT_NAME}/lib)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(WIN32)
    message(STATUS "Windows Build System.")
    message(STATUS "Teaserpp is not supported on Windows currently.")
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)

        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build Cpp examples" ON)

###############################################################################
# Find 3rdparty dependencies

find_package(Threads REQUIRED)

# Find OpenMP for parallel computing
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(3rdparty)

# Find open3d (Linux default)
set(Open3D_DIR "/usr/local/lib/cmake/Open3D" CACHE FILEPATH "Set to find Open3D")
find_package(Open3D REQUIRED NO_DEFAULT_PATH)

# Find pybind11 for Python wrappers
if(BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()

set(CMAKE_INSTALL_PREFIX $ENV{HOME} CACHE PATH "install path")
message(STATUS "Install path: " ${CMAKE_INSTALL_PREFIX})

include_directories(${SPDLOG_INCLUDE_DIRS})
add_subdirectory(src)

if(BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
    # install python examples 
    install(DIRECTORY examples/python
        DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/examples"
        USE_SOURCE_PERMISSIONS
        )
    install(DIRECTORY examples/data
    DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/examples"
    USE_SOURCE_PERMISSIONS
    )
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Coding style check 
if (BUILD_PYTHON)
    # `make check-style` checks style for c++/cuda/python/ipynb files
    add_custom_target(check-style
        COMMAND python3
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/check_style.py
        COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
    )

    # `make apply-style` applies style for c++/cuda/python/ipynb files
    add_custom_target(apply-style
        COMMAND python3
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/check_style.py --do_apply_style
        COMMENT "Python executable used for style check: ${Python3_EXECUTABLE}."
    )
endif()

# `make check-cpp-style` checks style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(check-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DDO_APPLY_STYLE=OFF
    -P ${CMAKE_CURRENT_SOURCE_DIR}/utils/check_cpp_style.cmake
)

# `make apply-cpp-style` applies style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(apply-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DDO_APPLY_STYLE=ON
    -P ${CMAKE_CURRENT_SOURCE_DIR}/utils/check_cpp_style.cmake
)
