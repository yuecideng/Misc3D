set(SRCS utils.cpp
         logging.cpp
         knn.cpp
         ransac.cpp
         filter.cpp
         normal_estimation.cpp
         proximity_extraction.cpp
         iterative_plane_segmentation.cpp
         boundary_detection.cpp
         transform_estimation.cpp
         correspondence_matching.cpp
         ppf_estimation.cpp
         ray_cast_renderer.cpp
         vis_utils.cpp)

if(BUILD_RECONSTRUCTION)
    list(APPEND SRCS pipeline.cpp)
endif()

add_library(misc3d ${SRCS})
target_include_directories(misc3d PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(misc3d PUBLIC 
    Open3D::Open3D  
    teaser_registration
    OpenMP::OpenMP_CXX
    Threads::Threads)
if(BUILD_RECONSTRUCTION)
  target_link_libraries(misc3d PUBLIC ${OpenCV_LIBS})
endif()

add_library(misc3d::misc3d ALIAS misc3d)

install(DIRECTORY ../include
  DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/"
  USE_SOURCE_PERMISSIONS
  )

install(TARGETS misc3d teaser_registration pmc
  EXPORT misc3d_target
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/lib"
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/lib"
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/bin"
)

install(EXPORT misc3d_target 
    NAMESPACE misc3d::
    FILE misc3d-config.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/misc3d/lib/cmake/)

