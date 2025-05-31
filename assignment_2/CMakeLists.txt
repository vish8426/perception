cmake_minimum_required(VERSION 2.8.3)
project(assignment_2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
	COMPONENTS
 	rospy 
 	interactive_markers
    pluginlib
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    moveit_visual_tools
    geometric_shapes
    tf2_ros
    tf2_eigen
    tf2_geometry_msgs
    )

## System dependencies are found with CMake's conventions
find_package(Eigen3 REQUIRED)

## catkin specific configuration
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES assignment_2
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)
