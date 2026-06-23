cmake_minimum_required(VERSION 3.8)
project(agv_4wis_gazebo_open)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(yaml-cpp REQUIRED)

add_library(agv_4wis_joint_controller SHARED src/joint_controller_plugin.cpp)
target_compile_features(agv_4wis_joint_controller PUBLIC cxx_std_17)
target_include_directories(agv_4wis_joint_controller PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_directories(agv_4wis_joint_controller PRIVATE ${GAZEBO_LIBRARY_DIRS})
target_link_libraries(agv_4wis_joint_controller ${GAZEBO_LIBRARIES} yaml-cpp)
ament_target_dependencies(agv_4wis_joint_controller
  ament_index_cpp gazebo_ros geometry_msgs nav_msgs rclcpp sensor_msgs std_msgs
  tf2_ros visualization_msgs
)

add_library(agv_hide_collisions_gui SHARED src/hide_collisions_gui_plugin.cpp)
target_compile_features(agv_hide_collisions_gui PUBLIC cxx_std_17)
target_include_directories(agv_hide_collisions_gui PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_directories(agv_hide_collisions_gui PRIVATE ${GAZEBO_LIBRARY_DIRS})
target_link_libraries(agv_hide_collisions_gui ${GAZEBO_LIBRARIES} Qt5::Widgets)

add_executable(double_ackermann_controller src/double_ackermann_controller.cpp)
target_compile_features(double_ackermann_controller PUBLIC cxx_std_17)
ament_target_dependencies(double_ackermann_controller geometry_msgs rclcpp std_msgs)

add_executable(gamepad_teleop src/gamepad_teleop.cpp)
target_compile_features(gamepad_teleop PUBLIC cxx_std_17)
ament_target_dependencies(gamepad_teleop rclcpp sensor_msgs std_msgs)

add_executable(pointcloud_merger src/pointcloud_merger.cpp)
target_compile_features(pointcloud_merger PUBLIC cxx_std_17)
ament_target_dependencies(pointcloud_merger rclcpp sensor_msgs)

add_executable(pointcloud_to_scan src/pointcloud_to_scan.cpp)
target_compile_features(pointcloud_to_scan PUBLIC cxx_std_17)
ament_target_dependencies(pointcloud_to_scan geometry_msgs rclcpp sensor_msgs tf2_ros)

install(TARGETS
  agv_4wis_joint_controller
  agv_hide_collisions_gui
  double_ackermann_controller
  gamepad_teleop
  pointcloud_merger
  pointcloud_to_scan
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config launch models rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
