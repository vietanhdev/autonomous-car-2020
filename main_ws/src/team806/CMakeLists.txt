cmake_minimum_required(VERSION 2.8.3)
project(team806)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
)

set(OpenCV_STATIC OFF)
find_package(  OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp
  DEPENDS system_lib
)

include_directories(
  include
  src/yaml-cpp/include
  include/lane_detector
  include/obstacle_detector
  include/traffic_sign
  include/smart_control
  include/object_detector
  ${catkin_INCLUDE_DIRS}
)

# yaml-cpp library
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "disable yaml tests")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "disable yaml tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "disable yaml contrib")
add_subdirectory(src/yaml-cpp)

add_executable(${PROJECT_NAME}_node
  src/racecar.cpp
  src/config.cpp
  src/timer.cpp
  src/smart_control/car_control.cpp
  src/smart_control/PID.cpp
  src/lane_detector/lane_detector.cpp
  src/traffic_sign/traffic_sign.cpp
  src/traffic_sign/traffic_sign_detector.cpp
  src/traffic_sign/traffic_sign_detector_2.cpp
  src/object_detector/detected_object.cpp
  src/object_detector/hog_based_object_detector.cpp
  src/object_detector/object_detector_manager.cpp
  src/object_detector/templ_matching_object_detector.cpp

)

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

target_link_libraries(
  ${PROJECT_NAME}_node
  yaml-cpp
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
)


message(${catkin_LIBRARIES})

install(TARGETS ${ADDITIONAL_EXECUTABLES}
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)
