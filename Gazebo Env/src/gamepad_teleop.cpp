#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float64.hpp>

namespace agv_4wis_gazebo
{

class GamepadTeleop : public rclcpp::Node
{
public:
  GamepadTeleop()
  : Node("gamepad_teleop")
  {
    axis_steer_ = declare_parameter("axis_steer", 0);
    axis_speed_ = declare_parameter("axis_speed", 1);
    button_enable_ = declare_parameter("button_enable", 0);
    button_crab_ = declare_parameter("button_crab", 4);
    button_spin_ = declare_parameter("button_spin", 5);
    max_steering_angle_ = declare_parameter("max_steering_angle", 0.65);
    max_wheel_speed_ = declare_parameter("max_wheel_speed", 12.0);
    crab_angle_ = declare_parameter("crab_angle", 1.57079632679);
    spin_angle_ = declare_parameter("spin_angle", 1.1200);
    deadzone_ = declare_parameter("deadzone", 0.08);
    command_timeout_ = declare_parameter("command_timeout", 0.30);
    invert_speed_axis_ = declare_parameter("invert_speed_axis", false);
    require_enable_button_ = declare_parameter("require_enable_button", true);
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

    joy_subscription_ = create_subscription<sensor_msgs::msg::Joy>(
      "joy", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::Joy::SharedPtr message) {
        on_joy(*message);
      });
    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(50),
      [this]() {
        if ((now() - last_joy_time_).seconds() > command_timeout_) {
          publish_stop(false);
        }
      });
    last_joy_time_ = now();
    publish_stop(true);

    RCLCPP_INFO(
      get_logger(),
      "Gamepad ready: hold button %d; LB crab, RB spin", button_enable_);
  }

private:
  static double shaped_axis(double value, double deadzone)
  {
    if (std::abs(value) <= deadzone) {
      return 0.0;
    }
    const double magnitude = (std::abs(value) - deadzone) / (1.0 - deadzone);
    return std::copysign(std::clamp(magnitude, 0.0, 1.0), value);
  }

  static bool pressed(const sensor_msgs::msg::Joy & joy, int index)
  {
    return index >= 0 && static_cast<std::size_t>(index) < joy.buttons.size() &&
           joy.buttons[index] != 0;
  }

  static double axis(const sensor_msgs::msg::Joy & joy, int index)
  {
    if (index < 0 || static_cast<std::size_t>(index) >= joy.axes.size()) {
      return 0.0;
    }
    return joy.axes[index];
  }

  void on_joy(const sensor_msgs::msg::Joy & joy)
  {
    last_joy_time_ = now();
    stopped_ = false;
    if (require_enable_button_ && !pressed(joy, button_enable_)) {
      publish_stop(false);
      return;
    }

    const double steer_input = shaped_axis(axis(joy, axis_steer_), deadzone_);
    double speed_input = shaped_axis(axis(joy, axis_speed_), deadzone_);
    if (invert_speed_axis_) {
      speed_input = -speed_input;
    }

    std::array<double, 4> steering{};
    std::array<double, 4> speeds{};

    if (pressed(joy, button_spin_)) {
      // Order: rear-right, rear-left, front-right, front-left.
      steering = {-spin_angle_, spin_angle_, spin_angle_, -spin_angle_};
      speeds = {
        steer_input * max_wheel_speed_,
        -steer_input * max_wheel_speed_,
        steer_input * max_wheel_speed_,
        -steer_input * max_wheel_speed_};
    } else if (pressed(joy, button_crab_)) {
      const double direction = steer_input == 0.0 ? 0.0 :
        std::copysign(crab_angle_, steer_input);
      steering.fill(direction);
      speeds.fill(std::abs(steer_input) * max_wheel_speed_);
    } else {
      const double angle = steer_input * max_steering_angle_;
      // Counter-phase four-wheel steering.
      steering = {-angle, -angle, angle, angle};
      speeds.fill(speed_input * max_wheel_speed_);
    }

    publish(steering, speeds);
  }

  void publish(
    const std::array<double, 4> & steering,
    const std::array<double, 4> & speeds)
  {
    for (std::size_t i = 0; i < 4; ++i) {
      std_msgs::msg::Float64 steering_message;
      steering_message.data = steering[i];
      steering_publishers_[i]->publish(steering_message);

      std_msgs::msg::Float64 speed_message;
      const double sign = i < drive_signs_.size() ? drive_signs_[i] : 1.0;
      speed_message.data = speeds[i] * sign;
      drive_publishers_[i]->publish(speed_message);
    }
  }

  void publish_stop(bool force)
  {
    if (stopped_ && !force) {
      return;
    }
    const std::array<double, 4> zero{};
    publish(zero, zero);
    stopped_ = true;
  }

  int axis_steer_;
  int axis_speed_;
  int button_enable_;
  int button_crab_;
  int button_spin_;
  double max_steering_angle_;
  double max_wheel_speed_;
  double crab_angle_;
  double spin_angle_;
  double deadzone_;
  double command_timeout_;
  bool invert_speed_axis_;
  bool require_enable_button_;
  bool stopped_{true};
  std::vector<double> drive_signs_;
  rclcpp::Time last_joy_time_;

  std::array<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr, 4>
    steering_publishers_;
  std::array<rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr, 4>
    drive_publishers_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_subscription_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
};

}  // namespace agv_4wis_gazebo

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<agv_4wis_gazebo::GamepadTeleop>());
  rclcpp::shutdown();
  return 0;
}
