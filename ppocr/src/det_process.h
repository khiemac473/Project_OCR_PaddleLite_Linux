#pragma once
#include <string>
#include <vector>
#include <map>
#include "opencv2/core.hpp"
#include "paddle_api.h"

namespace ocr {

class DetProcess {
public:
    // Khởi tạo với đường dẫn mô hình, số luồng CPU và chế độ năng lượng (ví dụ: "LITE_POWER_HIGH")
    DetProcess(const std::string &model_path, int cpu_threads, const std::string &cpu_power_mode);

    // Hàm detect: chạy detection trên ảnh đầu vào với cấu hình config.
    // Trả về vector chứa các box (mỗi box là vector gồm 4 điểm [x, y] theo tọa độ ảnh gốc).
    std::vector<std::vector<std::vector<int>>> detect(const cv::Mat &img, const std::map<std::string, double> &config);

    // Accessors cho các thông số chuyển đổi (nếu cần cho recognition sau này)
    float getScale() const;
    int getPadLeft() const;
    int getPadTop() const;

private:
    // Hàm letterbox resize: đưa ảnh về kích thước target_size x target_size (640×640),
    // giữ tỷ lệ ban đầu và bổ sung padding đều.
    cv::Mat letterboxResize(const cv::Mat &img, int target_size, float &scale, int &pad_left, int &pad_top);

    // Hàm chuyển đổi dữ liệu ảnh từ định dạng NHWC sang NCHW và chuẩn hóa.
    void NHWC3ToNC3HW(const float* src, float* dst, int num_pixels,
                      const std::vector<float>& mean, const std::vector<float>& scale);

    // Tiến trình tiền xử lý: resize ảnh và chuẩn bị tensor cho mô hình.
    void Preprocess(const cv::Mat &srcimg, int target_size);

    // Hậu xử lý: lấy output của mô hình, xử lý bit_map và chuyển tọa độ về ảnh gốc.
    std::vector<std::vector<std::vector<int>>> Postprocess(const cv::Mat &srcimg,
                                                             const std::map<std::string, double> &config,
                                                             int det_db_use_dilate);

    // Predictor của Paddle Lite.
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
    // Ảnh sau letterbox resize (640×640)
    cv::Mat letterbox_img_;
    // Các thông số dùng để chuyển tọa độ từ không gian letterbox về ảnh gốc.
    float scale_ = 1.f;
    int pad_left_ = 0;
    int pad_top_ = 0;
};

} // namespace ocr
