#include "gimbal.hpp"

#include <fmt/format.h>

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  serial_.setBaudrate(921600);
  auto timeout = serial::Timeout::simpleTimeout(100);
  serial_.setTimeout(timeout);
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{ 
  auto mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));
  // tools::logger()->warn("[Gimbal] Sending command: mode={}, yaw={:.2f}, yaw_vel={:.2f}, yaw_acc={:.2f}, "
  //                        "pitch={:.2f}, pitch_vel={:.2f}, pitch_acc={:.2f}",
  //                         mode, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

bool Gimbal::read(uint8_t * buffer, size_t size)
{
  try {
    return serial_.read(buffer, size) == size;
  } catch (const std::exception & e) {
    // tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 500) {
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      error_count = 0;
      continue;
    }

    // 逐字节搜索帧头 'S','P'，避免因数据中出现 0x53 0x50 而帧同步错位
    if (!read(reinterpret_cast<uint8_t *>(&rx_data_.head[0]), 1)) {
      error_count++;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    if (rx_data_.head[0] != 'S') continue;

    if (!read(reinterpret_cast<uint8_t *>(&rx_data_.head[1]), 1)) {
      error_count++;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    if (rx_data_.head[1] != 'P') continue;

    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    // 数据校验：四元数模长 + 角度范围
    {
      float q0 = rx_data_.q[0], q1 = rx_data_.q[1], q2 = rx_data_.q[2], q3 = rx_data_.q[3];
      float norm = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
      if (norm < 0.9f || norm > 1.1f) {
        tools::logger()->warn("[Gimbal] Bad quaternion norm={:.4f}, dropping frame.", norm);
        continue;
      }
    }

    error_count = 0;
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    queue_.push({q, t});
    // 从四元数解算 yaw/pitch/roll (ZYX 内旋)
    Eigen::Vector3d q_ypr = tools::eulers(q, 2, 1, 0);
    double q_yaw = q_ypr[0] * 180.0 / M_PI;
    double q_pitch = q_ypr[1] * 180.0 / M_PI;
    double q_roll = q_ypr[2] * 180.0 / M_PI;

    std::lock_guard<std::mutex> lock(mutex_);

    state_.yaw = q_ypr[0];
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = q_ypr[1];
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    tools::logger()->info(
      "[Gimbal] yaw={:.2f}, pitch={:.2f} | Quat-> yaw={:.2f}, pitch={:.2f}, roll={:.2f} | "
      "q_xyzw=[{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
      state_.yaw, state_.pitch, q_yaw, q_pitch, q_roll,
      q.x(), q.y(), q.z(), q.w());
    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      // 等待云台设备稳定后再返回
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io