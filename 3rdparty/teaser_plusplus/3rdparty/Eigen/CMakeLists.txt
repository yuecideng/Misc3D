add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/eigen3/"
  "${CMAKE_CURRENT_LIST_DIR}/eigen3/unsupported/")

add_library(Eigen3::Eigen ALIAS Eigen3)

if (NOT TARGET Eigen3::Eigen)
  message(ERROR "Can not find eigen3!")
endif()

if(RVX_BUILD_TESTS)
  add_subdirectory(Tests)
endif(RVX_BUILD_TESTS)

