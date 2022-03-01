add_executable(farthest_point_sampling test_farthest_point_sampling.cpp)
target_link_libraries(farthest_point_sampling PUBLIC misc3d::misc3d)

install(TARGETS farthest_point_sampling 
                RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/bin")