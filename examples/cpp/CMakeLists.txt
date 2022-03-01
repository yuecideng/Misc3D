add_executable(farthest_point_sampling test_farthest_point_sampling.cpp)
target_link_libraries(farthest_point_sampling PUBLIC misc3d::misc3d)

add_executable(normal_estimation test_normal_estimation.cpp)
target_link_libraries(normal_estimation PUBLIC misc3d::misc3d)

add_executable(ppf_estimator test_ppf_estimator.cpp)
target_link_libraries(ppf_estimator PUBLIC misc3d::misc3d)

add_executable(preprocessing test_preprocessing.cpp)
target_link_libraries(preprocessing PUBLIC misc3d::misc3d)

add_executable(ransac_and_edge test_ransac_and_edge.cpp)
target_link_libraries(ransac_and_edge PUBLIC misc3d::misc3d)

add_executable(segmentation test_segmentation.cpp)
target_link_libraries(segmentation PUBLIC misc3d::misc3d)

add_executable(transform_estimation test_transform_estimation.cpp)
target_link_libraries(transform_estimation PUBLIC misc3d::misc3d)

install(TARGETS farthest_point_sampling normal_estimation ppf_estimator 
                preprocessing ransac_and_edge segmentation transform_estimation
                RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/bin")