#ifndef AUTO_AIM__YOLOS__NNDEPLOY_YOLO_HPP
#define AUTO_AIM__YOLOS__NNDEPLOY_YOLO_HPP

#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/nn/infantry_v8.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
// 使用 NNdeployment + Infantry-v8 模型的 YOLOBase 实现。
// 该层是把通用 infantry_v8::ArmorBoard 翻译成 auto_aim::Armor 的“薄 glue”，
// 不含任何推理/后处理细节；若需搬到其它工程，把本类替换成对应工程的翻译即可。
class NndeployYolo : public YOLOBase
{
public:
  NndeployYolo(const std::string & config_path, bool debug);

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1) override;

  // 不再使用“手工张量 → Armor”的后处理通道。
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  ArmorName name_from_id(int armor_id) const;

  std::unique_ptr<infantry_v8::Detector> detector_;
  int my_color_ = 1;  // 1=我方蓝色, 0=我方红色, 2=离线保留双方(不滤)

  bool debug_ = false;
  bool use_roi_ = false;
  cv::Rect roi_;
  cv::Point2f offset_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLOS__NNDEPLOY_YOLO_HPP
