message(STATUS "Build Python binding enable!")
set(PYSRCS pymisc3d.cpp)
           
pybind11_add_module(py_misc3d ${PYSRCS})
target_include_directories(py_misc3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(py_misc3d PRIVATE misc3d::misc3d)

set_target_properties(py_misc3d
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/python/misc3d"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/misc3d"
  )
set_target_properties(py_misc3d PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(py_misc3d PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

file(WRITE "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/misc3d/__init__.py"
  "from .py_misc3d import *\n"
  )

install(TARGETS py_misc3d
  EXPORT py_misc3d_target 
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/lib/python/misc3d")

install(FILES "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python/misc3d/__init__.py"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/misc3d/lib/python/misc3d")