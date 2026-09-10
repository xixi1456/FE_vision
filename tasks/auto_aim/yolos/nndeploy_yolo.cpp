#include "nndeploy_yolo.hpp"

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <utility>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
namespace
{
template <typename T>
T read_or(const YAML::Node & node, const char * key, T fallback)
{
  if (node[key] && node[key].IsDefined()) {
    try {
      return node[key].as<T>();
    } catch (const std::exception &) {
      // 字段存在但类型不对时回退到默认值，便于配置迁移。
    }
  }
  return fallback;
}
}  // namespace

NndeployYolo::NndeployYolo(const std::string & config_path, bool debug) : debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  // 友方过滤保留：由 enemy_color 反推 my_color（1=我方蓝色，0=我方红色）。
  // 需要离线演示(保留双方颜色/白色)时可在 yaml 中开 nn_keep_both: true。
  const std::string enemy_color = read_or<std::string>(yaml, "enemy_color", "red");
  const bool keep_both = read_or<bool>(yaml, "nn_keep_both", false);
  my_color_ = keep_both ? 2 : (enemy_color == "red" ? 1 : 0);

  const std::string model_path = read_or<std::string>(yaml, "nn_model_path", "");
  if (model_path.empty()) {
    throw std::runtime_error("[NndeployYolo] config key 'nn_model_path' is required!");
  }
  const std::string device = read_or<std::string>(yaml, "nn_device", "CPU");
  const float confidence = read_or<float>(yaml, "nn_conf_threshold", 0.5f);
  const std::string infer_mode = read_or<std::string>(yaml, "nn_infer_mode", "sync");

  detector_ = std::make_unique<infantry_v8::Detector>(
    model_path, device, confidence, infer_mode, "openvino", "auto_detect",
    DebugConfig(debug_, false));

  use_roi_ = read_or<bool>(yaml, "use_roi", false);
  const YAML::Node & roi_node = yaml["roi"];
  const int x = (roi_node && roi_node["x"]) ? roi_node["x"].as<int>() : 0;
  const int y = (roi_node && roi_node["y"]) ? roi_node["y"].as<int>() : 0;
  const int width = (roi_node && roi_node["width"]) ? roi_node["width"].as<int>() : 0;
  const int height = (roi_node && roi_node["height"]) ? roi_node["height"].as<int>() : 0;

  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  tools::logger()->info(
    "[NndeployYolo] Infantry-v8 initialized: my_color={}, model={}, device={}, conf={}",
    my_color_, model_path, device, confidence);
}

ArmorName NndeployYolo::name_from_id(int armor_id) const
{
  // Infantry-v8 类别定义：0哨兵 1英雄 2工程 3~5号步兵 6前哨站 7/8基地(底/顶)。
  switch (armor_id) {
    case 0: return ArmorName::sentry;
    case 1: return ArmorName::one;
    case 2: return ArmorName::two;
    case 3: return ArmorName::three;
    case 4: return ArmorName::four;
    case 5: return ArmorName::five;
    case 6: return ArmorName::outpost;
    case 7:
    case 8: return ArmorName::base;
    default: return ArmorName::not_armor;
  }
}

std::list<Armor> NndeployYolo::detect(const cv::Mat & raw_img, int frame_count)
{
  (void)frame_count;
  std::list<Armor> armors;
  if (raw_img.empty()) {
    tools::logger()->warn("[NndeployYolo] Empty img!, camera drop!");
    return armors;
  }

  cv::Mat input_img;
  if (use_roi_) {
    if (roi_.width == -1) roi_.width = raw_img.cols;
    if (roi_.height == -1) roi_.height = raw_img.rows;
    if (roi_.width <= 0 || roi_.height <= 0 || roi_.x < 0 || roi_.y < 0 ||
        roi_.x + roi_.width > raw_img.cols || roi_.y + roi_.height > raw_img.rows) {
      tools::logger()->warn("[NndeployYolo] Invalid ROI, skip this frame!");
      return armors;
    }
    input_img = raw_img(roi_);
  } else {
    input_img = raw_img;
  }

  const std::vector<infantry_v8::ArmorBoard> boards = detector_->detect(input_img, my_color_);

  for (const infantry_v8::ArmorBoard & board : boards) {
    if (board.corners.size() != 4) continue;

    cv::Rect box = cv::boundingRect(board.corners);
    const cv::Point2f offset = use_roi_ ? offset_ : cv::Point2f(0.f, 0.f);

    // 复用神经网络构造函数完成几何量(center/ratio/rect_error)初始化；
    // 颜色(color_id: 0蓝1红其它extinguish)语义与 Infantry-v8 契约一致。
    Armor armor(board.color_id, board.armor_id, static_cast<float>(board.score), box,
                board.corners, offset);

    // 构造函数里对 armor_id 的旧映射不适用于 v8 契约（例如 8 应视为基地顶部），
    // 这里按 v8 类别表显式覆盖；尺寸直接采用模型给出的 size。
    armor.name = name_from_id(board.armor_id);
    armor.type = board.size == 1 ? ArmorType::big : ArmorType::small;

    // box 属于 ROI 坐标系，偏移到原图像素坐标系与 points 保持一致。
    if (use_roi_) {
      armor.box.x += static_cast<int>(offset_.x);
      armor.box.y += static_cast<int>(offset_.y);
    }

    // 白色装甲板保留但“不打”：color_id==2 已由构造函数映射为 extinguish，
    // tracker 的 enemy_color 过滤会将其剔除，与旧行为一致。
    armors.emplace_back(std::move(armor));
  }

  if (debug_) {
    tools::logger()->info(
      "[NndeployYolo] frame_count={}, armor count={}", frame_count, armors.size());
    for (const Armor & armor : armors) {
      // 打印字符串而不是枚举整数，避免与模型 size(0=小 1=大) 的 0/1 语义混淆。
      // 注意：ArmorType 枚举为 { big=0, small=1 }。
      tools::logger()->info(
        "  name={}({}) color={}({}) type={}({}) conf={:.2f} center=({:.1f},{:.1f})",
        ARMOR_NAMES[armor.name], static_cast<int>(armor.name),
        COLORS[armor.color], static_cast<int>(armor.color),
        ARMOR_TYPES[armor.type], static_cast<int>(armor.type),
        armor.confidence, armor.center.x, armor.center.y);
    }

    // 与原 yolo(v5/v8/yolo11) 行为一致：debug 时弹窗显示检测画面，
    // camera_detect_test / auto_aim_debug_mpc 等只依赖 imshow 看到图像。
    cv::Mat detection = raw_img.clone();
    tools::draw_text(
      detection, fmt::format("nndeploy frame[{}]", frame_count), cv::Point(10, 30),
      cv::Scalar(255, 255, 255));
    for (const Armor & armor : armors) {
      const std::string info = fmt::format(
        "{:.2f} {} {}", armor.confidence, ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
      tools::draw_points(detection, armor.points, cv::Scalar(0, 255, 0));
      tools::draw_text(
        detection, info,
        cv::Point(cvRound(armor.center.x), cvRound(armor.center.y)), cv::Scalar(0, 255, 0), 0.6,
        1);
    }
    if (use_roi_) {
      cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
    }
    cv::resize(detection, detection, {}, 0.5, 0.5);
    cv::imshow("detection", detection);
  }

  return armors;
}

std::list<Armor> NndeployYolo::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  (void)scale;
  (void)output;
  (void)bgr_img;
  (void)frame_count;
  tools::logger()->warn("[NndeployYolo] postprocess() is not used! Use detect() instead.");
  return std::list<Armor>();
}

}  // namespace auto_aim
