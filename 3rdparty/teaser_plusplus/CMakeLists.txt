cmake_minimum_required(VERSION 3.10)
project(teaserpp VERSION 1.0.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/" ${CMAKE_MODULE_PATH})

# Check build types
# if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
#     message(STATUS "Setting build type to 'Release' as none was specified.")
#     set(CMAKE_BUILD_TYPE "Release" CACHE
#             STRING "Choose the type of build." FORCE)
# endif ()

# Options
option(BUILD_WITH_MARCH_NATIVE "Build with flag march=native" OFF)
option(ENABLE_MKL "Try to use Eigen with MKL" OFF)
option(ENABLE_DIAGNOSTIC_PRINT "Enable printing of diagnostic messages" OFF)

if (ENABLE_DIAGNOSTIC_PRINT)
    message(STATUS "Enable printing of diagnostic messages.")
    add_definitions(-DTEASER_DIAG_PRINT)
endif ()

# Find dependencies
# Eigen3
find_package(Eigen3 3.2 QUIET REQUIRED NO_MODULE)
if (ENABLE_MKL)
    find_package(MKL)
    if (MKL_FOUND)
        message(STATUS "MKL found at: ${MKL_LIBRARIES}")
        include_directories(${MKL_INCLUDE_DIR})
        add_definitions(-DEIGEN_USE_MKL_ALL)
        list(APPEND TEASERPP_BLAS_LAPACK_LIBS ${MKL_LIBRARIES})
    else ()
        message(STATUS "MKL not found.")
    endif ()
endif ()

# Building Targets
set(TEASERPP_ROOT ${CMAKE_CURRENT_LIST_DIR})
add_subdirectory(3rdparty)

set(SPECTRA_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spectra/include")
add_subdirectory(teaser)


