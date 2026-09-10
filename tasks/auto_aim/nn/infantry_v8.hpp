#ifndef AUTO_AIM__NN__INFANTRY_V8_HPP
#define AUTO_AIM__NN__INFANTRY_V8_HPP

// ---------------------------------------------------------------------------
// Infantry-v8 模型接入的通用适配层（与具体自瞄工程解耦）。
//
// 只依赖：
//   1. OpenCV（cv::Mat / cv::Point2d）
//   2. NNdeployment 部署库的公开接口 network_deployment_interface.hpp
//      （ArmorModel / NetArmorResult / JsonConfig / DebugConfig）
//
// 该文件不 include 任何 sp_vision 内部头文件，可整体拷贝到其他工程复用。
// 上层工程只需将 ArmorBoard 映射成自己的装甲板结构即可。
//
// 输出契约（与开源 Infantry-v8n 一致）：
//   color_id : 0=蓝, 1=红, 2=白, 3=紫
//   armor_id : 0=哨兵,1=英雄,2=工程,3/4/5=3/4/5号步兵,6=前哨站,7=基地(底部),8=基地(顶部)
//   size     : 0=小装甲板, 1=大装甲板
// ---------------------------------------------------------------------------

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#include "network_deployment_interface.hpp"

namespace infantry_v8
{
// 四个角点已按 [左上, 右上, 右下, 左下] 归一，可直接用于 PnP 解算。
struct ArmorBoard
{
  cv::Point2f center;
  std::vector<cv::Point2f> corners;  // tl, tr, br, bl（原图坐标，输入 cv::Mat 坐标系）
  int armor_id = -1;                 // 类别 ID，语义见文件头注释
  int color_id = -1;                 // 颜色 ID，语义见文件头注释
  int size = 0;                      // 0=小装甲板，1=大装甲板
  double score = 0.;
  std::string class_name;            // armor_id 对应类别名称
  std::string color_name;            // color_id 对应颜色名称
};

// ArmorModel 的轻量封装：负责推理、坐标还原、友方颜色过滤(在库内完成)。
class Detector
{
public:
  // model_path 为模型完整路径；OpenVINO 传 .xml，TensorRT 传 .engine/.trt。
  Detector(
    const std::string & model_path,
    const std::string & device = "CPU",
    float confidence_threshold = 0.5f,
    const std::string & infer_mode = "sync",
    const std::string & deploy_way = "openvino",
    const std::string & postprocess_mode = "auto_detect",
    const DebugConfig & debug_config = DebugConfig());

  // 从 NNdeployment 的 JSON 配置入口构造。
  explicit Detector(const JsonConfig & json_config, const DebugConfig & debug_config = DebugConfig());

  // 对 bgr 图做装甲板推理。
  // my_color：1=我方蓝色，0=我方红色，2=测试模式(保留双方/白色全部输出)。
  // 库内部已过滤友方颜色与紫色异常候选。
  std::vector<ArmorBoard> detect(const cv::Mat & bgr_img, int my_color);

private:
  std::unique_ptr<ArmorModel> model_;
};

}  // namespace infantry_v8

#endif  // AUTO_AIM__NN__INFANTRY_V8_HPP
