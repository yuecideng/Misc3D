set(SRCS utils.cpp
         ransac.cpp
         normal_estimation.cpp
         proximity_extraction.cpp)

add_library(misc3d SHARED ${SRCS})
# target_include_directories(misc3d PRIVATE ${SPDLOG_INCLUDE_DIRS})
target_include_directories(misc3d PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(misc3d PUBLIC
    Open3D::Open3D  
    OpenMP::OpenMP_CXX)

add_library(misc3d::misc3d ALIAS misc3d)

install(DIRECTORY ../include
  DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/"
  USE_SOURCE_PERMISSIONS
  )

install(TARGETS misc3d
        EXPORT misc3d_target
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/lib"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/bin"
  )

install(EXPORT misc3d_target
    NAMESPACE misc3d::
    FILE misc3d-config.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/misc3d/lib/cmake/)