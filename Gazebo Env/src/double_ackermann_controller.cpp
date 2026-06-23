#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

namespace agv_4wis_gazebo_open
{

class DoubleAckermannController : public rclcpp::Node
{
public:
  DoubleAckermannController()
  : Node("double_ackermann_controller")
  {
    wheelbase_ = declare_parameter("wheelbase", 5.40082);
    track_width_ = declare_parameter("track_width", 2.32517);
    wheel_radius_ = declare_parameter("wheel_radius", 0.14995);
    max_steering_angle_ = declare_parameter("max_steering_angle", 1.10);
    max_wheel_speed_ = declare_parameter("max_wheel_speed", 30.0);
    command_timeout_ = declare_parameter("command_timeout", 0.30);
    drive_signs_ = declare_parameter(
      "drive_signs", std::vector<double>{-1.0, -1.0, -1.0, -1.0});

    const std::array<std::string, 4> steering_names{
      "frame_wheel10", "frame_wheel20", "frame_wheel30", "frame_wheel40"};
    const std::array<std::string, 4> drive_names{
      "frame_wheel1", "frame_wheel2", "frame_wheel3", "frame_wheel4"};
    for (std::size_t i = 0; i < 4; ++i) {
      steering_publishers_[i] = create_publisher<std_msgs::msg::Float64>(
        "/agv_4wis/" + steering_names[i] + "/command", 10);
      drive_publishers_[i] = create_publisher<std_msgs::msg::Float64>(
        "/agv_4wis/" + drive_names[i] + "/command", 10);
    }

    command_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      [this](const geometry_msgs::msg::Twist::SharedPtr message) {
        publish_double_ackermann(message->linear.x, message->angular.z);
        last_command_time_ = now();
        stopped_ = false;
      });
    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      [this]() {
        if (!stopped_ && (now() - last_command_time_).seconds() > command_timeout_) {
          publish_stop();
        }
      });
    last_command_time_ = now();
    publish_stop();

    RCLCPP_INFO(
      get_logger(),
      "Double-Ackermann /cmd_vel controller ready: wheelbase=%.4f m, track=%.4f m",
      wheelbase_, track_width_);
  }

private:
  std::pair<double, double> wheel_command(
    double linear_speed, double yaw_rate, double x, double y) const
  {
    const double velocity_x = linear_speed - yaw_rate * y;
    const double velocity_y = yaw_rate * x;
    double steering = std::atan2(velocity_y, velocity_x);
    double wheel_speed = std::hypot(velocity_x, velocity_y) / wheel_radius_;

    // Use the equivalent wheel direction within the physical steering range.
    if (steering > M_PI_2) {
      steering -= M_PI;
      wheel_speed = -wheel_speed;
    } else if (steering < -M_PI_2) {
      steering += M_PI;
      wheel_speed = -wheel_speed;
    }

    steering = std::clamp(steering, -max_steering_angle_, max_steering_angle_);
    wheel_speed = std::clamp(wheel_speed, -max_wheel_speed_, max_wheel_speed_);
    return {steering, wheel_speed};
  }

  void publish_double_ackermann(double linear_speed, double yaw_rate)
  {
    const double half_length = 0.5 * wheelbase_;
    const double half_width = 0.5 * track_width_;
    // Joint order: rear-right, rear-left, front-right, front-left.
    const std::array<std::pair<double, double>, 4> positions{{
      {-half_length, -half_width},
      {-half_length, half_width},
      {half_length, -half_width},
      {half_length, half_width}}};

    std::array<double, 4> steering{};
    std::array<double, 4> wheel_speeds{};
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const auto command = wheel_command(
        linear_speed, yaw_rate, positions[i].first, positions[i].second);
      steering[i] = command.first;
      wheel_speeds[i] = command.second;
    }
    publish(steering, wheel_speeds);
  }

  void publish(
    const std::array<double, 4> & steering,
    const std::array<double, 4> & wheel_speeds)
  {
    for (std::size_t i = 0; i < 4; ++i) {
      std_msgs::msg::Float64 steering_message;
      steering_message.data = steering[i];
      steering_publishers_[i]->publish(steering_message);

      std_msgs::msg::Float64 speed_message;
      const double sign = i < drive_signs_.size() ? drive_signs_[i] : 1.0;
      speed_message.data = sign * wheel_speeds[i];
      drive_publishers_[i]->publish(speed_message);
    }
  }

  void publish_stop()
  {
    const std::array<double, 4> zero{};
    publish(zero, zero);
    stopped_ = true;
  }

  double wheelbase_;
  double track_width_;
  double wheel_radius_;
  double max_steering_angle_;
  double max_wheel_speed_;
  double command_timeout_;
  bool stopped_{true};
  std::vector<double> drive_signs_;
  rclcpp::Time last_command_time_;

  std::array<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr, 4>
    steering_publishers_;
  std::array<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr, 4>
    drive_publishers_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
};

}  // namespace agv_4wis_gazebo_open

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<agv_4wis_gazebo_open::DoubleAckermannController>());
  rclcpp::shutdown();
  return 0;
}
