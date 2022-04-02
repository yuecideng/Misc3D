if(NOT WIN32)
    add_subdirectory(teaser_plusplus)
endif()

# if(BUILD_WITH_OPEN3D)
#     message(STATUS "Build Open3D internal with Misc3D.")
#     option(STATIC_WINDOWS_RUNTIME  "Use static (MT/MTd) Windows runtime"  ON )
#     include(ExternalProject)
#     ExternalProject_Add(
#         external_open3d
#         PREFIX open3d
#         GIT_REPOSITORY https://github.com/intel-isl/Open3D.git
#         GIT_TAG master
#         GIT_SHALLOW ON
#         UPDATE_COMMAND ""
#         # Check out https://github.com/intel-isl/Open3D/blob/master/CMakeLists.txt
#         # For the full list of available options.
#         CMAKE_ARGS
#             -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#             -DGLIBCXX_USE_CXX11_ABI=ON
#             -DSTATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
#             -DBUILD_SHARED_LIBS=ON
#             -DBUILD_PYTHON_MODULE=OFF
#             -DBUILD_EXAMPLES=OFF
#     )

#     ExternalProject_Get_Property(external_open3d INSTALL_DIR)
#     add_library(Open3DHelper INTERFACE)
#     add_dependencies(Open3DHelper external_open3d)
#     target_compile_features(Open3DHelper INTERFACE cxx_std_14)
#     target_include_directories(Open3DHelper INTERFACE $<BUILD_INTERFACE:${INSTALL_DIR}/include> $<BUILD_INTERFACE:${INSTALL_DIR}/include/open3d/3rdparty>)
#     target_link_directories(Open3DHelper INTERFACE $<BUILD_INTERFACE:${INSTALL_DIR}/lib>)
#     target_link_libraries(Open3DHelper INTERFACE Open3D)
#     add_library(Open3D::Open3D ALIAS Open3DHelper)
# endif()
