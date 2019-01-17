cmake_minimum_required(VERSION 2.8.3)
project(ransac_slam)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
#set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
  image_transport
  cv_bridge
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(Eigen3 REQUIRED)
#find_package(OpenCV REQUIRED)
# 寻找OpenCV库
set(OpenCV_DIR "/usr/local/opencv320/share/OpenCV")
find_package( OpenCV 3.2 REQUIRED )


###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ransac_slam
  CATKIN_DEPENDS roscpp rospy std_msgs tf visualization_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/include
	${EIGEN3_INCLUDE_DIR}
)

## Declare a C++ library
add_library(${PROJECT_NAME} SHARED
src/System.cpp
src/Tracking.cpp
src/Map.cpp
src/ExtendKF.cpp
src/Converter.cpp
)
target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ${EIGEN3_LIBS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
## Specify libraries to link a library or executable target against
add_executable(mono_node examples/Monocular/mono_slam.cpp)
target_link_libraries(mono_node ${PROJECT_NAME})

add_executable(odometry test/odometry.cpp)
target_link_libraries(odometry ${PROJECT_NAME})

#############
## Install ##
#############


## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
   PATTERN ".svn" EXCLUDE
)
