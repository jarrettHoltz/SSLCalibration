PROJECT(delta_calibration)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(arch ${})

MESSAGE(STATUS "Compilers found: ${CMAKE_CXX_COMPILER_LIST}")
MESSAGE(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")
MESSAGE(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")

SET(CMAKE_CXX_FLAGS "-std=c++11 -march=native -Wall -g -rdynamic")

IF(${CMAKE_BUILD_TYPE} MATCHES "Release")
  MESSAGE(STATUS "Additional Flags for Release mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -O3 -DNDEBUG")
  SET(BUILD_SPECIFIC_LIBRARIES "tcmalloc")
ELSEIF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  MESSAGE(STATUS "Additional Flags for Debug mode")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -rdynamic")
  SET(BUILD_SPECIFIC_LIBRARIES "")
ENDIF()

# End Build Notes and flags

INCLUDE($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
rosbuild_init()
SET(ROS_BUILD_STATIC_LIBS true)
SET(ROS_BUILD_SHARED_LIBS false)
# End ROS things

MESSAGE(STATUS "ROS-Overrride Build Type: ${CMAKE_BUILD_TYPE}")
MESSAGE(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "Build-Specific Libraries: ${BUILD_SPECIFIC_LIBRARIES}")

# Executable path
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# QT stuff
SET(QT_USE_QTNETWORK true)
SET(QT_USE_QTOPENGL true)

OPTION(CALIBRATION_TARGETS "Enable complication of delta-calibration" FALSE)


find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
FIND_PACKAGE(Qt4 REQUIRED)
FIND_PACKAGE(Lua51 REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})


INCLUDE(${QT_USE_FILE})


INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/libraries)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/delta_calibration)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/shared/math)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/shared/util)
IF(CALIBRATION_TARGETS)
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/delta_calibration)
ENDIF(CALIBRATION_TARGETS)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

SET(libs roslib roscpp ncurses popt pthread  glut
     ${LUA_LIBRARIES} rt
    ${BUILD_SPECIFIC_LIBRARIES} glog rosbag X11)


SET(SHARED_SRCS
  src/shared/util/pthread_utils.cpp
  src/shared/util/configreader.cpp
)

SET(PERCEPTION_SRCS
  src/perception_tools/perception_2d.cpp
  #src/delta_calibration/icp.cpp
)

SET(FSPF_SRCS
  src/fspf/grahams_scan.cpp
  src/fspf/plane_filtering.cpp
  src/fspf/plane_polygon.cpp
)
ADD_LIBRARY(shared_libraries SHARED ${SHARED_SRCS} ${HW_SRCS})
ADD_DEPENDENCIES(shared_libraries rosbuild_precompile)
TARGET_LINK_LIBRARIES(shared_libraries shared_libraries ${libs})

ADD_LIBRARY(fspf_library ${FSPF_SRCS})
ADD_DEPENDENCIES(fspf_library shared_libraries)
TARGET_LINK_LIBRARIES(fspf_library shared_libraries ${libs})

# ADD_LIBRARY(perception_library ${PERCEPTION_SRCS})
# ADD_DEPENDENCIES(perception_library shared_libraries)
# TARGET_LINK_LIBRARIES(perception_library shared_libraries ${libs})