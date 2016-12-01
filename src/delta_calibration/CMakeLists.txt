set(CMAKE_AUTOMOC ON)

SET(target delta_calibration)
ROSBUILD_ADD_EXECUTABLE(${target}
    delta_calibration.cpp
    icp.cpp
    ../perception_tools/kdtree.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    shared_libraries
    fspf_library
    perception_library
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    shared_libraries 
    fspf_library
    ${OpenCV_LIBRARIES} ${libs})
    
SET(target ssl_calibration)
ROSBUILD_ADD_EXECUTABLE(${target}
    ssl_calibration.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    shared_libraries
    fspf_library
    perception_library
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    shared_libraries 
    fspf_library
    ${OpenCV_LIBRARIES} ${libs})
    
SET(target test_normals)
ROSBUILD_ADD_EXECUTABLE(${target}
    test_normals.cpp
    icp.cpp
    ../perception_tools/kdtree.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    shared_libraries
    fspf_library
    perception_library
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    shared_libraries 
    fspf_library
    ${OpenCV_LIBRARIES} ${libs})
    
SET(target visualizes_poses)
ROSBUILD_ADD_EXECUTABLE(${target}
    visualize_poses.cpp
    icp.cpp
    ../perception_tools/kdtree.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    shared_libraries
    fspf_library
    perception_library
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    shared_libraries 
    fspf_library
    ${OpenCV_LIBRARIES} ${libs})
    
 SET(target find_z)
ROSBUILD_ADD_EXECUTABLE(${target}
    find_z.cpp
    icp.cpp
    ../perception_tools/kdtree.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    shared_libraries
    fspf_library
    perception_library
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    shared_libraries 
    fspf_library
    ${OpenCV_LIBRARIES} ${libs})
    
SET(target check_transform)
ROSBUILD_ADD_EXECUTABLE(${target} check_transform.cpp
                                  icp.cpp
                                  ../perception_tools/kdtree.cpp)
ADD_DEPENDENCIES(${target}
    ceres
    shared_libraries
    fspf_library
    ${OpenCV_LIBRARIES})
TARGET_LINK_LIBRARIES(${target}
    ceres
    shared_libraries 
    fspf_library
    ${OpenCV_LIBRARIES} ${libs})
