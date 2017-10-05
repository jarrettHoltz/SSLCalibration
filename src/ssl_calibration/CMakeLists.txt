SET(target ssl_calibration)
ROSBUILD_ADD_EXECUTABLE(${target}
    ssl_calibration.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    ${OpenCV_LIBRARIES} ${libs})