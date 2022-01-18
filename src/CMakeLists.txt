set(SRCS Segmentation.cpp
         PrimitivesFitting.cpp)

add_library(primitives_fitting_lib SHARED ${SRCS})
target_include_directories(primitives_fitting_lib PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(primitives_fitting_lib PUBLIC
    Open3D::Open3D
    OpenMP::OpenMP_CXX)

add_library(primitives_fitting_lib::primitives_fitting_lib ALIAS primitives_fitting_lib)

install(DIRECTORY ../include
  DESTINATION "${CMAKE_INSTALL_PREFIX}/PrimitivesFittingLib/"
  USE_SOURCE_PERMISSIONS
  )

install(TARGETS primitives_fitting_lib
        EXPORT primitives_fitting_lib_target
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/PrimitivesFittingLib/lib"
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/PrimitivesFittingLib/bin"
  )

install(EXPORT primitives_fitting_lib_target
    NAMESPACE primitives_fitting_lib::
    FILE primitives_fitting_lib-config.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/PrimitivesFittingLib/lib/cmake/)