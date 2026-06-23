#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

using namespace std::chrono_literals;

class PointCloudMerger : public rclcpp::Node
{
public:
  PointCloudMerger()
  : Node("pointcloud_merger")
  {
    input_topics_ = declare_parameter<std::vector<std::string>>(
      "input_topics",
      {
        "/agv_4wis/lidar/rear_right/points",
        "/agv_4wis/lidar/rear_left/points",
        "/agv_4wis/lidar/front_right/points",
        "/agv_4wis/lidar/front_left/points",
      });
    sensor_x_ = declare_parameter<std::vector<double>>("sensor_x");
    sensor_y_ = declare_parameter<std::vector<double>>("sensor_y");
    sensor_z_ = declare_parameter<std::vector<double>>("sensor_z");
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/agv_4wis/lidar/points_merged");
    output_frame_ = declare_parameter<std::string>("output_frame", "frame");
    publish_rate_ = declare_parameter<double>("publish_rate", 10.0);
    max_cloud_age_ = declare_parameter<double>("max_cloud_age", 0.5);
    filter_center_x_ = declare_parameter<double>("filter_center_x", 0.0);
    filter_center_y_ = declare_parameter<double>("filter_center_y", 0.0);
    filter_size_x_ = declare_parameter<double>("filter_size_x", 6.0);
    filter_size_y_ = declare_parameter<double>("filter_size_y", 3.0);
    filter_min_z_ = declare_parameter<double>("filter_min_z", 0.05);
    filter_max_z_ = declare_parameter<double>("filter_max_z", 3.0);
    filter_min_range_ = declare_parameter<double>("filter_min_range", 0.20);

    const auto count = input_topics_.size();
    if (count == 0 || sensor_x_.size() != count || sensor_y_.size() != count ||
      sensor_z_.size() != count)
    {
      throw std::runtime_error(
              "input_topics, sensor_x, sensor_y and sensor_z must have equal non-zero lengths");
    }
    if (publish_rate_ <= 0.0 || filter_size_x_ < 0.0 || filter_size_y_ < 0.0 ||
      filter_min_z_ > filter_max_z_ || filter_min_range_ < 0.0)
    {
      throw std::runtime_error(
              "publish_rate/filter sizes/Z filtering limits are invalid");
    }

    clouds_.resize(count);
    subscriptions_.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
      subscriptions_.push_back(create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topics_[index], rclcpp::SensorDataQoS(),
        [this, index](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
          std::lock_guard<std::mutex> lock(mutex_);
          clouds_[index].message = std::move(message);
          clouds_[index].received = std::chrono::steady_clock::now();
        }));
    }

    // Sensor data must stay real-time: keep only the latest sample rather
    // than queueing stale clouds when a consumer briefly falls behind.
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS().keep_last(1));
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_),
      std::bind(&PointCloudMerger::publish_merged, this));

    RCLCPP_INFO(
      get_logger(),
      "Merging %zu lidar clouds to %s; Z range=[%.3f, %.3f], "
      "XY exclusion center=(%.3f, %.3f), size=(%.3f, %.3f)",
      count, output_topic_.c_str(), filter_min_z_, filter_max_z_,
      filter_center_x_, filter_center_y_,
      filter_size_x_, filter_size_y_);
  }

private:
  struct CloudSlot
  {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr message;
    std::chrono::steady_clock::time_point received;
  };

  void publish_merged()
  {
    std::vector<CloudSlot> clouds;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      clouds = clouds_;
    }

    const auto current_time = std::chrono::steady_clock::now();
    std::vector<float> xyz;
    for (std::size_t index = 0; index < clouds.size(); ++index) {
      const auto & slot = clouds[index];
      if (!slot.message) {
        continue;
      }
      const double age = std::chrono::duration<double>(current_time - slot.received).count();
      if (max_cloud_age_ > 0.0 && age > max_cloud_age_) {
        continue;
      }

      try {
        sensor_msgs::PointCloud2ConstIterator<float> x(*slot.message, "x");
        sensor_msgs::PointCloud2ConstIterator<float> y(*slot.message, "y");
        sensor_msgs::PointCloud2ConstIterator<float> z(*slot.message, "z");
        for (; x != x.end(); ++x, ++y, ++z) {
          const float frame_x = *x + static_cast<float>(sensor_x_[index]);
          const float frame_y = *y + static_cast<float>(sensor_y_[index]);
          const float frame_z = *z + static_cast<float>(sensor_z_[index]);
          if (!std::isfinite(frame_x) || !std::isfinite(frame_y) ||
            !std::isfinite(frame_z))
          {
            continue;
          }
          const float local_range_squared = *x * *x + *y * *y + *z * *z;
          if (local_range_squared <
            static_cast<float>(filter_min_range_ * filter_min_range_))
          {
            continue;
          }
          if (frame_z < filter_min_z_ || frame_z > filter_max_z_) {
            continue;
          }
          const bool inside_vehicle =
            std::abs(frame_x - filter_center_x_) <= filter_size_x_ * 0.5 &&
            std::abs(frame_y - filter_center_y_) <= filter_size_y_ * 0.5;
          if (!inside_vehicle) {
            xyz.push_back(frame_x);
            xyz.push_back(frame_y);
            xyz.push_back(frame_z);
          }
        }
      } catch (const std::runtime_error & error) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Skipping malformed cloud on %s: %s",
          input_topics_[index].c_str(), error.what());
      }
    }

    sensor_msgs::msg::PointCloud2 output;
    output.header.stamp = now();
    output.header.frame_id = output_frame_;
    output.height = 1;
    output.is_dense = true;
    output.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier modifier(output);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(xyz.size() / 3);

    sensor_msgs::PointCloud2Iterator<float> out_x(output, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(output, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(output, "z");
    for (std::size_t index = 0; index < xyz.size(); index += 3, ++out_x, ++out_y, ++out_z) {
      *out_x = xyz[index];
      *out_y = xyz[index + 1];
      *out_z = xyz[index + 2];
    }
    publisher_->publish(output);
  }

  std::vector<std::string> input_topics_;
  std::vector<double> sensor_x_;
  std::vector<double> sensor_y_;
  std::vector<double> sensor_z_;
  std::string output_topic_;
  std::string output_frame_;
  double publish_rate_;
  double max_cloud_age_;
  double filter_center_x_;
  double filter_center_y_;
  double filter_size_x_;
  double filter_size_y_;
  double filter_min_z_;
  double filter_max_z_;
  double filter_min_range_;

  std::mutex mutex_;
  std::vector<CloudSlot> clouds_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subscriptions_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PointCloudMerger>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("pointcloud_merger"), "%s", error.what());
  }
  rclcpp::shutdown();
  return 0;
}
