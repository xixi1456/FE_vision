#include <fmt/core.h>

#include <chrono>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace
{
void draw_armor(cv::Mat & img, const auto_aim::Armor & armor)
{
  const std::string label = fmt::format(
    "{:.2f} {} {} d={:.2f}m", armor.confidence, auto_aim::ARMOR_NAMES[armor.name],
    auto_aim::ARMOR_TYPES[armor.type], armor.xyz_in_gimbal.norm());
  std::vector<cv::Point> polygon;
  for (const cv::Point2f & p : armor.points) {
    polygon.emplace_back(cvRound(p.x), cvRound(p.y));
  }
  cv::polylines(img, polygon, true, cv::Scalar(0, 255, 0), 2);
  cv::putText(
    img, label, cv::Point(cvRound(armor.center.x), cvRound(armor.center.y)),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
}
}  // namespace

// usage: nndeploy_video_check <video> [config] [output.mp4]
int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::fprintf(
      stderr, "usage: %s <video> [config] [output.mp4]\n", argc > 0 ? argv[0] : "check");
    return 1;
  }
  const std::string video_path = argv[1];
  const std::string config_path =
    argc > 2 ? argv[2] : std::string("configs/nndeploy_standard3.yaml");
  const std::string output_path = argc > 3 ? argv[3] : std::string();

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::fprintf(stderr, "无法打开视频: %s\n", video_path.c_str());
    return 1;
  }
  const double fps = cap.get(cv::CAP_PROP_FPS);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);

  cv::VideoWriter writer;
  if (!output_path.empty()) {
    const cv::Size size(
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    writer.open(
      output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
    if (!writer.isOpened()) {
      std::fprintf(stderr, "无法创建输出视频: %s\n", output_path.c_str());
      return 1;
    }
  }

  cv::Mat frame;
  long frame_index = 0;
  long detected_frames = 0;
  double sum_ms = 0.;

  while (cap.read(frame)) {
    const auto t0 = std::chrono::steady_clock::now();
    auto armors = yolo.detect(frame);
    const auto dt_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    sum_ms += dt_ms;
    ++frame_index;

    std::string summary;
    for (auto & armor : armors) {
      solver.solve(armor);
      draw_armor(frame, armor);
      summary += fmt::format(
        "  [{} {:.2f} dist={:.2f}]", auto_aim::ARMOR_NAMES[armor.name], armor.confidence,
        armor.xyz_in_gimbal.norm());
    }
    if (!armors.empty()) ++detected_frames;
    if (!armors.empty() && frame_index % 10 == 1) {
      std::printf(
        "#%ld detect=%zu (%.1fms):%s\n", frame_index, armors.size(), dt_ms,
        summary.c_str());
    }

    if (writer.isOpened()) writer.write(frame);
    if (frame_index % 30 == 0) {
      std::printf("  ... processed %ld frames (avg %.1fms, detected %ld/%ld)\n",
                  frame_index, sum_ms / frame_index, detected_frames, frame_index);
    }
  }

  std::printf(
    "done: %ld frames, avg %.1f ms/frame, detected ratio %.1f%%\n", frame_index,
    sum_ms / std::max(1L, frame_index),
    100.0 * static_cast<double>(detected_frames) / std::max(1L, frame_index));
  return 0;
}
