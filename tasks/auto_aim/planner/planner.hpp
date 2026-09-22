#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <array>
#include <list>
#include <mutex>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  std::array<float, HORIZON> predicted_yaw{};
};

class Planner
{
public:
  Planner(const std::string & config_path);

  Eigen::Vector4d get_debug_xyza() const;

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double comming_angle_, leaving_angle_;
  int lock_slot_id_ = -1;  // 锁定模式：防止在两个都呈45度的装甲板之间来回切换

  Eigen::Vector4d debug_xyza_;
  mutable std::mutex debug_mutex_;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void select_target_xyza(
    const Target & target, const std::vector<Eigen::Vector4d> & armor_xyza_list, Eigen::Vector3d & xyz,
    double & yaw);

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP