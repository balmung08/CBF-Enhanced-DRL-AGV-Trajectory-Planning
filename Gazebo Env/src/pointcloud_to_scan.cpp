#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

class PointCloudToScan : public rclcpp::Node
{
public:
  PointCloudToScan()
  : Node("pointcloud_to_scan")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/agv_4wis/lidar/points_merged");
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/agv_4wis/scan");
    output_frame_ = declare_parameter<std::string>("output_frame", "frame");
    parent_frame_ = declare_parameter<std::string>("parent_frame", "frame");
    scan_z_ = declare_parameter<double>("scan_z", 1.13);
    min_height_ = declare_parameter<double>("min_height", 0.10);
    max_height_ = declare_parameter<double>("max_height", 2.00);
    angle_min_ = declare_parameter<double>("angle_min", -M_PI);
    angle_max_ = declare_parameter<double>("angle_max", M_PI);
    angle_increment_ = declare_parameter<double>(
      "angle_increment", M_PI / 180.0);
    range_min_ = declare_parameter<double>("range_min", 0.20);
    range_max_ = declare_parameter<double>("range_max", 30.0);
    scan_time_ = declare_parameter<double>("scan_time", 0.10);
    use_inf_ = declare_parameter<bool>("use_inf", true);
    inf_epsilon_ = declare_parameter<double>("inf_epsilon", 1.0);

    if (min_height_ > max_height_ || angle_min_ >= angle_max_ ||
      angle_increment_ <= 0.0 || range_min_ < 0.0 ||
      range_min_ >= range_max_ || scan_time_ < 0.0)
    {
      throw std::runtime_error("Invalid PointCloud2-to-LaserScan parameters");
    }

    publisher_ = create_publisher<sensor_msgs::msg::LaserScan>(
      output_topic_, rclcpp::SensorDataQoS().keep_last(1));
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&PointCloudToScan::cloud_callback, this, std::placeholders::_1));

    static_tf_broadcaster_ =
      std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = now();
    transform.header.frame_id = parent_frame_;
    transform.child_frame_id = output_frame_;
    transform.transform.translation.z = scan_z_;
    transform.transform.rotation.w = 1.0;
    static_tf_broadcaster_->sendTransform(transform);

    RCLCPP_INFO(
      get_logger(),
      "Converting %s to %s: height=[%.2f, %.2f] m, angle increment=%.3f deg",
      input_topic_.c_str(), output_topic_.c_str(), min_height_, max_height_,
      angle_increment_ * 180.0 / M_PI);
  }

private:
  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    sensor_msgs::msg::LaserScan scan;
    scan.header = cloud->header;
    scan.header.frame_id = output_frame_;
    scan.angle_min = static_cast<float>(angle_min_);
    scan.angle_max = static_cast<float>(angle_max_);
    scan.angle_increment = static_cast<float>(angle_increment_);
    scan.scan_time = static_cast<float>(scan_time_);
    scan.range_min = static_cast<float>(range_min_);
    scan.range_max = static_cast<float>(range_max_);

    const auto beam_count = static_cast<std::size_t>(
      std::floor((angle_max_ - angle_min_) / angle_increment_)) + 1U;
    // The complete scan is projected from one PointCloud2 snapshot. It is
    // not acquired beam-by-beam, so per-beam timing would make RViz wait for
    // future TF samples and visually lag behind the vehicle.
    scan.time_increment = 0.0F;
    const float empty_value = use_inf_ ?
      std::numeric_limits<float>::infinity() :
      static_cast<float>(range_max_ + inf_epsilon_);
    scan.ranges.assign(beam_count, empty_value);

    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*cloud, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*cloud, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*cloud, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z) ||
          *z < min_height_ || *z > max_height_)
        {
          continue;
        }

        const double range_squared =
          static_cast<double>(*x) * *x + static_cast<double>(*y) * *y;
        if (range_squared < range_min_ * range_min_ ||
          range_squared > range_max_ * range_max_)
        {
          continue;
        }

        const double angle = std::atan2(*y, *x);
        if (angle < angle_min_ || angle > angle_max_) {
          continue;
        }
        const auto index = static_cast<std::size_t>(
          std::floor((angle - angle_min_) / angle_increment_));
        if (index >= scan.ranges.size()) {
          continue;
        }
        const float range = static_cast<float>(std::sqrt(range_squared));
        scan.ranges[index] = std::min(scan.ranges[index], range);
      }
    } catch (const std::runtime_error & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Skipping malformed cloud: %s", error.what());
      return;
    }

    publisher_->publish(scan);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string output_frame_;
  std::string parent_frame_;
  double scan_z_;
  double min_height_;
  double max_height_;
  double angle_min_;
  double angle_max_;
  double angle_increment_;
  double range_min_;
  double range_max_;
  double scan_time_;
  bool use_inf_;
  double inf_epsilon_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr publisher_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PointCloudToScan>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("pointcloud_to_scan"), "%s", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
