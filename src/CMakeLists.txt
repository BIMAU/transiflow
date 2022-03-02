cmake_minimum_required(VERSION 3.15.0)
project(FVM CXX)

find_package(MPI)
if (MPI_FOUND)
  set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})

  # For now, just include everything
  include_directories (${MPI_CXX_INCLUDE_PATH})
  include_directories (${MPI_INCLUDE_PATH})
else()
  message(WARNING "could not find MPI. Trying to compile anyway, presuming the compiler/linker knows where to find it.")
endif()

# Disable some assert that causes conversions to base classes to fail
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

find_package(SWIG)
find_package(PyTrilinos)
find_package(ShyLU_DDFROSch)

# FIXME: Remove HYMLS dependecy once we don't need the solver anymore
find_package(HYMLS)

if (SWIG_FOUND AND PyTrilinos_FOUND AND ShyLU_DDFROSch_FOUND AND HYMLS_FOUND)
  if (POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
  endif ()
  if (POLICY CMP0086)
    cmake_policy(SET CMP0086 NEW)
  endif ()

  find_package(Python COMPONENTS Interpreter Development)
  include_directories(${Python_INCLUDE_DIRS})
  include_directories(${PyTrilinos_INCLUDE_DIRS})
  include_directories(${ShyLU_DDFROSch_INCLUDE_DIRS})
  include_directories(${ShyLU_DDFROSch_TPL_INCLUDE_DIRS})
  include_directories(${HYMLS_INCLUDE_DIRS})

  if (NOT ${Python_VERSION} VERSION_LESS 3.0)
    list(APPEND CMAKE_SWIG_FLAGS -py3)
  endif ()

  include(${SWIG_USE_FILE})

  include_directories(BEFORE ${PROJECT_SOURCE_DIR})

  set(SOURCE_FILE FROSch.i)

  set_property(SOURCE ${SOURCE_FILE} PROPERTY CPLUSPLUS ON)
  swig_add_library(FROSch LANGUAGE Python SOURCES ${SOURCE_FILE})
  swig_link_libraries(FROSch ${Python_LIBRARIES})
  swig_link_libraries(FROSch ${PyTrilinos_LIBRARIES})
  swig_link_libraries(FROSch ${ShyLU_DDFROSch_LIBRARIES})
  swig_link_libraries(FROSch ${HYMLS_LIBRARIES})

  set(SITE_PACKAGE_DIR "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/FROSch")
  install(TARGETS FROSch DESTINATION ${SITE_PACKAGE_DIR})
  get_property(SUPPORT_FILES TARGET FROSch PROPERTY SWIG_SUPPORT_FILES)
  install(FILES ${SUPPORT_FILES} DESTINATION ${SITE_PACKAGE_DIR})
  install(FILES init.py DESTINATION ${SITE_PACKAGE_DIR} RENAME __init__.py)
endif ()
