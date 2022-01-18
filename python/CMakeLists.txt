message(STATUS "Build Python binding enable!")
set(PYSRCS ransac.cpp 
           segmentation.cpp
           py_primitives_fitting.cpp)
           
pybind11_add_module(py_primitives_fitting ${PYSRCS})
target_include_directories(py_primitives_fitting PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(py_primitives_fitting PUBLIC primitives_fitting_lib::primitives_fitting_lib Open3D::Open3D)

set_target_properties(py_primitives_fitting
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/python/primitives_fitting"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/primitives_fitting"
  )
set_target_properties(py_primitives_fitting PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(py_primitives_fitting PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

file(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/primitives_fitting/__init__.py"
  "from .py_primitives_fitting import *\n"
  )

install(TARGETS py_primitives_fitting
  EXPORT py_primitives_fitting_target 
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/PrimitivesFittingLib/lib/python/primitives_fitting")

install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/primitives_fitting/__init__.py"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/PrimitivesFittingLib/lib/python/primitives_fitting")