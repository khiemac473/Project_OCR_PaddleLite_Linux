#include <iostream>
#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "db_post_process.h"

namespace fs = std::filesystem;
using namespace paddle::lite_api;
// Hàm resize ảnh sao cho kích thước (chiều cao và chiều rộng)
// là bội số của 32 (theo mẫu code của bạn) và lưu lại tỷ lệ resize.
cv::Mat DetResizeImg(const cv::Mat img, int max_size_len, std::vector<float> &ratio_hw) {
  int w = img.cols;
  int h = img.rows;
  float ratio = 1.f;
  int max_wh = (w >= h ? w : h);
  if (max_wh > max_size_len) {
    if (h > w) {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
    } else {
      ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
    }
  }
  int resize_h = static_cast<int>(float(h) * ratio);
  int resize_w = static_cast<int>(float(w) * ratio);
  if (resize_h % 32 == 0)
    resize_h = resize_h;
  else if (resize_h / 32 < 1 + 1e-5)
    resize_h = 32;
  else
    resize_h = (resize_h / 32 - 1) * 32;
  if (resize_w % 32 == 0)
    resize_w = resize_w;
  else if (resize_w / 32 < 1 + 1e-5)
    resize_w = 32;
  else
    resize_w = (resize_w / 32 - 1) * 32;
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return resize_img;
}
// Hàm chuyển đổi dữ liệu ảnh từ NHWC (3 kênh) sang NCHW (3 kênh) và chuẩn hóa.
void NHWC3ToNC3HW(const float* src, float* dst, int num_pixels,
                  const std::vector<float>& mean, const std::vector<float>& scale) {
  for (int i = 0; i < num_pixels; i++) {
    for (int c = 0; c < 3; c++) {
      float value = src[i * 3 + c];
      value = (value - mean[c]) * scale[c];
      dst[c * num_pixels + i] = value;
    }
  }
}
// Hàm chuyển chuỗi chế độ năng lượng sang enum của Paddle Lite.
PowerMode ParsePowerMode(const std::string &mode) {
  if (mode == "LITE_POWER_HIGH") return LITE_POWER_HIGH;
  if (mode == "LITE_POWER_LOW")  return LITE_POWER_LOW;
  if (mode == "LITE_POWER_FULL") return LITE_POWER_FULL;
  return LITE_POWER_HIGH;
}
// Hàm crop vùng chữ từ ảnh dựa vào 4 điểm (box) thông qua biến đổi perspective.
cv::Mat CropBox(const cv::Mat &src, const std::vector<std::vector<int>> &box) {
  std::vector<cv::Point2f> src_pts;
  for (const auto &pt : box) {
    src_pts.push_back(cv::Point2f(static_cast<float>(pt[0]), static_cast<float>(pt[1])));
  }
  // Tính toán kích thước crop (chiều rộng và chiều cao)
  float widthA = std::hypot(src_pts[0].x - src_pts[1].x, src_pts[0].y - src_pts[1].y);
  float widthB = std::hypot(src_pts[2].x - src_pts[3].x, src_pts[2].y - src_pts[3].y);
  float maxWidth = std::max(widthA, widthB);
  
  float heightA = std::hypot(src_pts[0].x - src_pts[3].x, src_pts[0].y - src_pts[3].y);
  float heightB = std::hypot(src_pts[1].x - src_pts[2].x, src_pts[1].y - src_pts[2].y);
  float maxHeight = std::max(heightA, heightB);
  
  std::vector<cv::Point2f> dst_pts = {
    cv::Point2f(0, 0),
    cv::Point2f(maxWidth - 1, 0),
    cv::Point2f(maxWidth - 1, maxHeight - 1),
    cv::Point2f(0, maxHeight - 1)
  };
  
  cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
  cv::Mat cropped;
  cv::warpPerspective(src, cropped, M, cv::Size(static_cast<int>(maxWidth), static_cast<int>(maxHeight)));
  return cropped;
}
// Lớp DetPredictor: bao gồm tiền xử lý, chạy mô hình và hậu xử lý theo mẫu code của bạn.
class DetPredictor {
public:
  explicit DetPredictor(const std::string &modelDir, const int cpuThreadNum,
                        const std::string &cpuPowerMode) {
    MobileConfig config;
    config.set_model_from_file(modelDir);
    config.set_threads(cpuThreadNum);
    config.set_power_mode(ParsePowerMode(cpuPowerMode));
    predictor_ = CreatePaddlePredictor<MobileConfig>(config);
  }
  // Tiền xử lý: resize ảnh và chuẩn bị dữ liệu cho mô hình.
  void Preprocess(const cv::Mat &srcimg, const int max_side_len) {
    cv::Mat img = DetResizeImg(srcimg, max_side_len, ratio_hw_);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);
    
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor_->GetInput(0)));
    input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor0->mutable_data<float>();
    
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    NHWC3ToNC3HW(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
  }
  
  // Hậu xử lý: sử dụng output của mô hình để trích xuất các box vùng chữ.
  // Các hàm BoxesFromBitmap và FilterTagDetRes được dùng từ db_post_process.h.
  std::vector<std::vector<std::vector<int>>>
  Postprocess(const cv::Mat srcimg, std::map<std::string, double> Config,
              int det_db_use_dilate) {
    std::unique_ptr<const Tensor> output_tensor(
        std::move(predictor_->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();
    
    int out_size = shape_out[2] * shape_out[3];
    std::vector<float> pred(out_size);
    std::vector<unsigned char> cbuf(out_size);
    for (int i = 0; i < out_size; i++) {
      pred[i] = outptr[i];
      cbuf[i] = static_cast<unsigned char>(outptr[i] * 255);
    }
    
    cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1, cbuf.data());
    cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F, pred.data());
    
    const double threshold = double(Config["det_db_thresh"]) * 255;
    const double max_value = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
    if (det_db_use_dilate == 1) {
      cv::Mat dilation_map;
      cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
      cv::dilate(bit_map, dilation_map, dila_ele);
      bit_map = dilation_map;
    }
    auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);
    std::vector<std::vector<std::vector<int>>> filter_boxes =
        FilterTagDetRes(boxes, ratio_hw_[0], ratio_hw_[1], srcimg);
    
    return filter_boxes;
  }
  
  // Hàm Predict: chạy toàn bộ quy trình (tiền xử lý, chạy mô hình, hậu xử lý).
  std::vector<std::vector<std::vector<int>>>
  Predict(cv::Mat &img, std::map<std::string, double> Config) {
    cv::Mat srcimg;
    img.copyTo(srcimg);
    
    int max_side_len = int(Config["max_side_len"]);
    int det_db_use_dilate = int(Config["det_db_use_dilate"]);
    
    Preprocess(img, max_side_len);
    predictor_->Run();
    auto filter_boxes = Postprocess(srcimg, Config, det_db_use_dilate);
    return filter_boxes;
  }
  
private:
  std::vector<float> ratio_hw_;
  std::shared_ptr<PaddlePredictor> predictor_;
};

int main() {
  std::string model_path = "../models/model.nb";
  std::string input_dir = "../input";
  std::string output_dir = "../output";
  // Kiểm tra tồn tại thư mục input
  if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
    std::cerr << "Thư mục input không tồn tại: " << input_dir << std::endl;
    return -1;
  }
  fs::create_directories(output_dir);
  // Cấu hình cho detector
  std::map<std::string, double> Config;
  Config["max_side_len"] = 480;
  Config["det_db_thresh"] = 0.7;
  Config["det_db_unclip_ratio"] = 1.0;
  Config["det_db_use_dilate"] = 1;
  
  int cpuThreadNum = 4;
  std::string cpuPowerMode = "LITE_POWER_HIGH";
  DetPredictor detector(model_path, cpuThreadNum, cpuPowerMode);
  
  // Duyệt tất cả ảnh trong thư mục input (giả sử ảnh có đuôi jpg, png)
  int totalCrop = 0;
  for (const auto &entry : fs::directory_iterator(input_dir)) {
    if (!entry.is_regular_file()) continue;
    std::string file_path = entry.path().string();
    // Kiểm tra đuôi file (jpg, png, jpeg)
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
    cv::Mat image = cv::imread(file_path);
    if (image.empty()) {
      std::cerr << "Không thể tải ảnh: " << file_path << std::endl;
      continue;
    }
    
    auto boxes = detector.Predict(image, Config);
    
    // Với mỗi box (một box là vector 4 điểm), crop vùng chữ và lưu ra file.
    int idx = 0;
    for (const auto &box : boxes) {
      cv::Mat crop = CropBox(image, box);
      std::string output_path = output_dir + "/crop_" + std::to_string(totalCrop) + "_" + std::to_string(idx) + ".jpg";
      cv::imwrite(output_path, crop);
      idx++;
    }
    totalCrop += idx;
    std::cout << "Đã xử lý ảnh: " << file_path << " -> " << idx << " vùng chữ." << std::endl;
  }
  
  std::cout << "Tổng cộng đã cắt và lưu " << totalCrop << " vùng ảnh chứa chữ." << std::endl;
  return 0;
}
