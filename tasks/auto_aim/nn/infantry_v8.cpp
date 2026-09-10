#include "infantry_v8.hpp"

#include <utility>

namespace infantry_v8
{
namespace
{
ArmorBoard toArmorBoard(const NetArmorResult & result)
{
  ArmorBoard board;
  board.armor_id = result.armor_id;
  board.color_id = result.color_id;
  board.size = result.size;
  board.score = result.score;
  board.class_name = result.class_name;
  board.color_name = result.color_name;

  // NetArmorResult.points 顺序为 [左上, 左下, 右下, 右上]，
  // 这里归一为工程常用的 [左上, 右上, 右下, 左下]，保证 PnP 结果一致。
  if (result.points.size() == 4) {
    board.corners = {
      cv::Point2f(result.points[0]), cv::Point2f(result.points[3]),
      cv::Point2f(result.points[2]), cv::Point2f(result.points[1])};
    board.center =
      (board.corners[0] + board.corners[1] + board.corners[2] + board.corners[3]) / 4.f;
  }
  return board;
}
}  // namespace

Detector::Detector(
  const std::string & model_path,
  const std::string & device,
  float confidence_threshold,
  const std::string & infer_mode,
  const std::string & deploy_way,
  const std::string & postprocess_mode,
  const DebugConfig & debug_config)
: model_(std::make_unique<ArmorModel>(
    model_path, infer_mode, deploy_way, device, confidence_threshold, postprocess_mode,
    debug_config))
{
}

Detector::Detector(const JsonConfig & json_config, const DebugConfig & debug_config)
: model_(std::make_unique<ArmorModel>(json_config, debug_config))
{
}

std::vector<ArmorBoard> Detector::detect(const cv::Mat & bgr_img, int my_color)
{
  std::vector<ArmorBoard> boards;
  if (!model_) return boards;

  std::vector<NetArmorResult> results = model_->netProcess(bgr_img, my_color);
  boards.reserve(results.size());
  for (const NetArmorResult & result : results) {
    ArmorBoard board = toArmorBoard(result);
    if (board.corners.size() != 4) continue;
    boards.emplace_back(std::move(board));
  }
  return boards;
}

}  // namespace infantry_v8
