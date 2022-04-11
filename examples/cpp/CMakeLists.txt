add_executable(farthest_point_sampling farthest_point_sampling.cpp)
target_link_libraries(farthest_point_sampling PUBLIC misc3d::misc3d)

add_executable(normal_estimation normal_estimation.cpp)
target_link_libraries(normal_estimation PUBLIC misc3d::misc3d)

add_executable(ppf_estimator ppf_estimator.cpp)
target_link_libraries(ppf_estimator PUBLIC misc3d::misc3d)

add_executable(ray_cast_rendering ray_cast_rendering.cpp)
target_link_libraries(ray_cast_rendering PUBLIC misc3d::misc3d)

add_executable(preprocessing preprocessing.cpp)
target_link_libraries(preprocessing PUBLIC misc3d::misc3d)

add_executable(ransac_and_boundary ransac_and_boundary.cpp)
target_link_libraries(ransac_and_boundary PUBLIC misc3d::misc3d)

add_executable(segmentation segmentation.cpp)
target_link_libraries(segmentation PUBLIC misc3d::misc3d)

add_executable(segment_plane_iterative segment_plane_iterative.cpp)
target_link_libraries(segment_plane_iterative PUBLIC misc3d::misc3d)

add_executable(transform_estimation transform_estimation.cpp)
target_link_libraries(transform_estimation PUBLIC misc3d::misc3d)

install(TARGETS farthest_point_sampling normal_estimation ppf_estimator ray_cast_rendering
                preprocessing ransac_and_boundary segmentation segment_plane_iterative 
                transform_estimation
                RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/bin")

if(WIN32)
    # Copy Open3D.dll to build/bin for runtime dynamic linking
    file(COPY ${Open3D_INCLUDE_DIRS}/../bin/Open3D.dll DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
    # Install Open3D.dll to misc3d/bin for runtime dynamic linking
    install(FILES ${Open3D_INCLUDE_DIRS}/../bin/Open3D.dll DESTINATION misc3d/bin)
endif()
