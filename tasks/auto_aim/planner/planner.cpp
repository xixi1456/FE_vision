#include "planner.hpp"

#include <algorithm>
#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  comming_angle_ =
    yaml["comming_angle"] ? yaml["comming_angle"].as<double>() / 57.3 : 45.0 / 57.3;  // degree to rad
  leaving_angle_ =
    yaml["leaving_angle"] ? yaml["leaving_angle"].as<double>() / 57.3 : 20.0 / 57.3;  // degree to rad

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Eigen::Vector4d Planner::get_debug_xyza() const
{
  std::lock_guard<std::mutex> lock(debug_mutex_);
  return debug_xyza_;
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 2 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);
  for (int i = 0; i < HORIZON; i++) {
    plan.predicted_yaw[i] = tools::limit_rad(traj(0, i) + yaw0);
  }

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

void Planner::select_target_xyza(
  const Target & target, const std::vector<Eigen::Vector4d> & armor_xyza_list, Eigen::Vector3d & xyz,
  double & yaw)
{
  auto min_dist = 1e10;
  auto armor_num = armor_xyza_list.size();

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(target.ekf_x()[2], target.ekf_x()[0]);

  // 不考虑小陀螺（状态定义 [x vx y vy z vz a w r l h]：x[7] 是角速度 w，x[8] 是旋转半径 r）
  if (std::abs(target.ekf_x()[7]) <= 2 && target.name != ArmorName::outpost) {
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    std::vector<double> delta_angle_list;
    for (int i = 0; i < armor_num; i++) {
      auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
      if (std::abs(delta_angle) > 60 / 57.3) continue;
      id_list.push_back(i);
      delta_angle_list.push_back(delta_angle);
    }

    if (!id_list.empty()) {
      // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
      if (id_list.size() > 1) {
        // 锁定的装甲板仍在可射击范围内时，保持锁定
        if (std::find(id_list.begin(), id_list.end(), lock_slot_id_) == id_list.end()) {
          // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
          lock_slot_id_ = id_list.front();
          auto min_abs_delta = std::abs(delta_angle_list.front());
          for (std::size_t j = 1; j < id_list.size(); j++) {
            if (std::abs(delta_angle_list[j]) < min_abs_delta) {
              min_abs_delta = std::abs(delta_angle_list[j]);
              lock_slot_id_ = id_list[j];
            }
          }
        }
        xyz = armor_xyza_list[lock_slot_id_].head<3>();
        yaw = armor_xyza_list[lock_slot_id_][3];
        return;
      }

      // 只有一个装甲板在可射击范围内时，退出锁定模式
      lock_slot_id_ = -1;
      xyz = armor_xyza_list[id_list[0]].head<3>();
      yaw = armor_xyza_list[id_list[0]][3];
      return;
    }
  } else {
    // 小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
    Eigen::VectorXd ekf_x = target.ekf_x();
    for (int i = 0; i < armor_num; i++) {
      auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
      if (std::abs(delta_angle) > comming_angle_) continue;
      if (ekf_x[7] > 0 && delta_angle < leaving_angle_) {
        xyz = armor_xyza_list[i].head<3>();
        yaw = armor_xyza_list[i][3];
        return;
      }
      if (ekf_x[7] < 0 && delta_angle > -leaving_angle_) {
        xyz = armor_xyza_list[i].head<3>();
        yaw = armor_xyza_list[i][3];
        return;
      }
    }
  }

  // 无可射击板：选择 xy 距离最近的装甲板
  for (const auto & xyza : armor_xyza_list) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  select_target_xyza(target, target.armor_xyza_list(), xyz, yaw);

  {
    std::lock_guard<std::mutex> lock(debug_mutex_);
    debug_xyza_ = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);
  }

  auto dist = xyz.head<2>().norm();
  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim