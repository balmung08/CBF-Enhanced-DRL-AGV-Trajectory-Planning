#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>
#include <yaml-cpp/yaml.h>

namespace agv_4wis_gazebo
{

class JointControllerPlugin : public gazebo::ModelPlugin
{
public:
  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = std::move(model);
    node_ = gazebo_ros::Node::Get(sdf);

    const auto package_name =
      sdf->HasElement("config_package") ?
      sdf->Get<std::string>("config_package") : "agv_4wis_gazebo";
    const auto share = ament_index_cpp::get_package_share_directory(package_name);

    auto files = sdf->GetElement("joint_config");
    while (files) {
      load_joint_config(share + "/" + files->Get<std::string>());
      files = files->GetNextElement("joint_config");
    }

    if (controllers_.size() != 8U) {
      RCLCPP_ERROR(
        node_->get_logger(), "Expected 8 joint configurations, loaded %zu",
        controllers_.size());
    }

    joint_state_publisher_ =
      node_->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    odometry_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("/agv_4wis/odom", 10);
    local_odometry_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("/agv_4wis/odom_local", 10);
    footprint_publisher_ = node_->create_publisher<visualization_msgs::msg::Marker>(
      "/agv_4wis/vehicle_footprint", rclcpp::QoS(1).transient_local());
    transform_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

    initial_pose_ = model_->WorldPose();
    publish_vehicle_footprint();

    last_update_time_ = model_->GetWorld()->SimTime();
    last_odometry_time_ = last_update_time_;
    last_footprint_time_ = last_update_time_;
    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      std::bind(&JointControllerPlugin::on_update, this));

    RCLCPP_INFO(
      node_->get_logger(),
      "Loaded independent ROS 2 control for %zu AGV joints; world origin=(%.3f, %.3f, %.3f)",
      controllers_.size(),
      initial_pose_.Pos().X(), initial_pose_.Pos().Y(), initial_pose_.Pos().Z());
  }

private:
  enum class Mode { Position, Velocity };

  struct Controller
  {
    std::string joint_name;
    std::string command_topic;
    Mode mode{Mode::Position};
    gazebo::physics::JointPtr joint;
    double command{0.0};
    double command_min{-1.0e16};
    double command_max{1.0e16};
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr subscription;
  };

  void load_joint_config(const std::string & path)
  {
    const auto yaml = YAML::LoadFile(path);
    const auto cfg = yaml["joint"];

    Controller controller;
    controller.joint_name = cfg["name"].as<std::string>();
    controller.command_topic = cfg["command_topic"].as<std::string>();
    controller.mode =
      cfg["control_mode"].as<std::string>() == "velocity" ?
      Mode::Velocity : Mode::Position;
    controller.command = cfg["initial_command"].as<double>(0.0);
    controller.command_min = cfg["command_min"].as<double>(-1.0e16);
    controller.command_max = cfg["command_max"].as<double>(1.0e16);
    controller.joint = model_->GetJoint(controller.joint_name);
    if (!controller.joint) {
      throw std::runtime_error("Gazebo joint not found: " + controller.joint_name);
    }

    controllers_.push_back(std::move(controller));
    auto & stored = controllers_.back();
    stored.subscription = node_->create_subscription<std_msgs::msg::Float64>(
      stored.command_topic, 10,
      [this, index = controllers_.size() - 1](const std_msgs::msg::Float64::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        auto & item = controllers_.at(index);
        item.command = std::clamp(msg->data, item.command_min, item.command_max);
      });

    RCLCPP_INFO(
      node_->get_logger(), "%s: %s control on %s",
      stored.joint_name.c_str(),
      stored.mode == Mode::Position ? "position" : "velocity",
      stored.command_topic.c_str());
  }

  void on_update()
  {
    const auto now = model_->GetWorld()->SimTime();
    const double dt = (now - last_update_time_).Double();
    if (dt <= 0.0) {
      return;
    }
    last_update_time_ = now;

    sensor_msgs::msg::JointState state;
    state.header.stamp = node_->now();

    std::lock_guard<std::mutex> lock(command_mutex_);
    for (auto & controller : controllers_) {
      const double position = controller.joint->Position(0);
      if (controller.mode == Mode::Position) {
        controller.joint->SetPosition(0, controller.command, true);
      } else {
        controller.joint->SetVelocity(0, controller.command);
      }

      state.name.push_back(controller.joint_name);
      state.position.push_back(
        controller.mode == Mode::Position ? controller.command : position);
      state.velocity.push_back(
        controller.mode == Mode::Velocity ? controller.command : 0.0);
      state.effort.push_back(0.0);
    }
    joint_state_publisher_->publish(state);

    if ((now - last_odometry_time_).Double() >= 0.01) {
      publish_odometry();
      last_odometry_time_ = now;
    }
    if ((now - last_footprint_time_).Double() >= (1.0 / 30.0)) {
      publish_vehicle_footprint();
      last_footprint_time_ = now;
    }
  }

  void publish_odometry()
  {
    const auto pose = model_->WorldPose();
    const auto linear_world = model_->WorldLinearVel();
    const auto angular_world = model_->WorldAngularVel();
    const auto rotation_inverse = pose.Rot().Inverse();
    const auto linear_body = rotation_inverse.RotateVector(linear_world);
    const auto angular_body = rotation_inverse.RotateVector(angular_world);

    const auto stamp = node_->now();

    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = "odom";
    odometry.child_frame_id = "frame";
    odometry.pose.pose.position.x = pose.Pos().X();
    odometry.pose.pose.position.y = pose.Pos().Y();
    odometry.pose.pose.position.z = pose.Pos().Z();
    odometry.pose.pose.orientation.x = pose.Rot().X();
    odometry.pose.pose.orientation.y = pose.Rot().Y();
    odometry.pose.pose.orientation.z = pose.Rot().Z();
    odometry.pose.pose.orientation.w = pose.Rot().W();
    odometry.twist.twist.linear.x = linear_body.X();
    odometry.twist.twist.linear.y = linear_body.Y();
    odometry.twist.twist.linear.z = linear_body.Z();
    odometry.twist.twist.angular.x = angular_body.X();
    odometry.twist.twist.angular.y = angular_body.Y();
    odometry.twist.twist.angular.z = angular_body.Z();
    odometry_publisher_->publish(odometry);

    const auto initial_rotation_inverse = initial_pose_.Rot().Inverse();
    const auto local_position =
      initial_rotation_inverse.RotateVector(pose.Pos() - initial_pose_.Pos());
    const auto local_rotation = initial_rotation_inverse * pose.Rot();

    nav_msgs::msg::Odometry local_odometry = odometry;
    local_odometry.header.frame_id = "world";
    local_odometry.pose.pose.position.x = local_position.X();
    local_odometry.pose.pose.position.y = local_position.Y();
    local_odometry.pose.pose.position.z = local_position.Z();
    local_odometry.pose.pose.orientation.x = local_rotation.X();
    local_odometry.pose.pose.orientation.y = local_rotation.Y();
    local_odometry.pose.pose.orientation.z = local_rotation.Z();
    local_odometry.pose.pose.orientation.w = local_rotation.W();
    local_odometry_publisher_->publish(local_odometry);

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = "world";
    transform.child_frame_id = "frame";
    transform.transform.translation.x = local_position.X();
    transform.transform.translation.y = local_position.Y();
    transform.transform.translation.z = local_position.Z();
    transform.transform.rotation = local_odometry.pose.pose.orientation;
    transform_broadcaster_->sendTransform(transform);
  }

  void publish_vehicle_footprint()
  {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = node_->now();
    marker.header.frame_id = "frame";
    marker.ns = "agv_4wis";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.frame_locked = true;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.10;
    marker.color.r = 0.1F;
    marker.color.g = 0.9F;
    marker.color.b = 0.3F;
    marker.color.a = 1.0F;

    constexpr double half_length = 2.96;
    constexpr double half_width = 1.46;
    constexpr double height = 0.08;
    for (const auto & corner : std::array<std::pair<double, double>, 5>{{
        {half_length, half_width},
        {half_length, -half_width},
        {-half_length, -half_width},
        {-half_length, half_width},
        {half_length, half_width}}})
    {
      geometry_msgs::msg::Point point;
      point.x = corner.first;
      point.y = corner.second;
      point.z = height;
      marker.points.push_back(point);
    }
    footprint_publisher_->publish(marker);
  }

  gazebo::physics::ModelPtr model_;
  gazebo_ros::Node::SharedPtr node_;
  gazebo::event::ConnectionPtr update_connection_;
  gazebo::common::Time last_update_time_;
  gazebo::common::Time last_odometry_time_;
  gazebo::common::Time last_footprint_time_;
  ignition::math::Pose3d initial_pose_;
  std::vector<Controller> controllers_;
  std::mutex command_mutex_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr local_odometry_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr footprint_publisher_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
};

GZ_REGISTER_MODEL_PLUGIN(JointControllerPlugin)

}  // namespace agv_4wis_gazebo
