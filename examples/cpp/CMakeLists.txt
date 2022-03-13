add_executable(farthest_point_sampling farthest_point_sampling.cpp)
target_link_libraries(farthest_point_sampling PUBLIC misc3d::misc3d)

add_executable(normal_estimation normal_estimation.cpp)
target_link_libraries(normal_estimation PUBLIC misc3d::misc3d)

add_executable(ppf_estimator ppf_estimator.cpp)
target_link_libraries(ppf_estimator PUBLIC misc3d::misc3d)

add_executable(preprocessing preprocessing.cpp)
target_link_libraries(preprocessing PUBLIC misc3d::misc3d)

add_executable(ransac_and_edge ransac_and_edge.cpp)
target_link_libraries(ransac_and_edge PUBLIC misc3d::misc3d)

add_executable(segmentation segmentation.cpp)
target_link_libraries(segmentation PUBLIC misc3d::misc3d)

add_executable(transform_estimation transform_estimation.cpp)
target_link_libraries(transform_estimation PUBLIC misc3d::misc3d)

install(TARGETS farthest_point_sampling normal_estimation ppf_estimator 
                preprocessing ransac_and_edge segmentation transform_estimation
                RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/bin")