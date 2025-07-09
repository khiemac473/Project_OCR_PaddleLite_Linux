#include "det_process.h"
#include "db_post_process.h"  // Chứa BoxesFromBitmap, FilterTagDetRes,...
#include "opencv2/imgproc.hpp"  // Để sử dụng cv::resize, cv::copyMakeBorder, cv::threshold, cv::polylines, cv::boundingRect,...
#include <algorithm>
#include <iostream>

// Bao gồm rec_process để sử dụng recognition.
#include "rec_process.h"

namespace ocr {
using namespace paddle::lite_api;

cv::Mat CropBox(const cv::Mat &src, const std::vector<std::vector<int>> &box) {
    std::vector<cv::Point2f> src_pts;
    for (const auto &pt : box) {
        src_pts.push_back(cv::Point2f(static_cast<float>(pt[0]), static_cast<float>(pt[1])));
    }
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

DetProcess::DetProcess(const std::string &model_path, int cpu_threads, const std::string &cpu_power_mode) {
    MobileConfig config;
    config.set_model_from_file(model_path);
    config.set_threads(cpu_threads);
    if (cpu_power_mode == "LITE_POWER_LOW")
        config.set_power_mode(LITE_POWER_LOW);
    else if (cpu_power_mode == "LITE_POWER_FULL")
        config.set_power_mode(LITE_POWER_FULL);
    else
        config.set_power_mode(LITE_POWER_HIGH);
    predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

cv::Mat DetProcess::letterboxResize(const cv::Mat &img, int target_size, float &scale, int &pad_left, int &pad_top) {
    int orig_w = img.cols;
    int orig_h = img.rows;
    scale = static_cast<float>(target_size) / std::max(orig_w, orig_h);
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    
    int pad_w = target_size - new_w;
    int pad_h = target_size - new_h;
    pad_left = pad_w / 2;
    pad_top  = pad_h / 2;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

void DetProcess::NHWC3ToNC3HW(const float* src, float* dst, int num_pixels,
                              const std::vector<float>& mean, const std::vector<float>& scale) {
    for (int i = 0; i < num_pixels; i++) {
        for (int c = 0; c < 3; c++) {
            float value = src[i * 3 + c];
            value = (value - mean[c]) * scale[c];
            dst[c * num_pixels + i] = value;
        }
    }
}

void DetProcess::Preprocess(const cv::Mat &srcimg, int target_size) {
    letterbox_img_ = letterboxResize(srcimg, target_size, scale_, pad_left_, pad_top_);
    cv::Mat img_fp;
    letterbox_img_.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);
    
    std::unique_ptr<Tensor> input_tensor(std::move(predictor_->GetInput(0)));
    input_tensor->Resize({1, 3, target_size, target_size});
    auto *data0 = input_tensor->mutable_data<float>();
    
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale_vec = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
    NHWC3ToNC3HW(dimg, data0, target_size * target_size, mean, scale_vec);
}

std::vector<std::vector<std::vector<int>>> DetProcess::Postprocess(const cv::Mat &srcimg,
                                                                     const std::map<std::string, double> &config,
                                                                     int det_db_use_dilate) {
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor_->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape = output_tensor->shape();
    int out_size = shape[2] * shape[3];
    std::vector<float> pred(out_size);
    std::vector<unsigned char> cbuf(out_size);
    for (int i = 0; i < out_size; i++) {
        pred[i] = outptr[i];
        cbuf[i] = static_cast<unsigned char>(outptr[i] * 255);
    }
    cv::Mat cbuf_map(shape[2], shape[3], CV_8UC1, cbuf.data());
    cv::Mat pred_map(shape[2], shape[3], CV_32F, pred.data());
    
    double threshold = config.at("det_db_thresh") * 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, 255, cv::THRESH_BINARY);
    if (det_db_use_dilate == 1) {
        cv::Mat dilation_map;
        cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, dilation_map, dila_ele);
        bit_map = dilation_map;
    }
    auto boxes = BoxesFromBitmap(pred_map, bit_map, config);
    auto filter_boxes = FilterTagDetRes(boxes, scale_, scale_, srcimg);
    return filter_boxes;
}

std::vector<std::vector<std::vector<int>>> DetProcess::detect(const cv::Mat &img, const std::map<std::string, double> &config) {
    cv::Mat srcimg;
    img.copyTo(srcimg);
    int target_size = static_cast<int>(config.at("max_side_len")); // target_size = 640
    int det_db_use_dilate = static_cast<int>(config.at("det_db_use_dilate"));
    
    Preprocess(img, target_size);
    predictor_->Run();
    auto boxes = Postprocess(srcimg, config, det_db_use_dilate);
    return boxes;
}

float DetProcess::getScale() const { return scale_; }
int DetProcess::getPadLeft() const { return pad_left_; }
int DetProcess::getPadTop() const { return pad_top_; }

} // namespace ocr

// ---------------- Main Function (Detection + Recognition Pipeline) ----------------

#include "rec_process.h" // Sử dụng lớp RecProcess để nhận dạng

int main() {
    using namespace ocr;
    // Đường dẫn mô hình detection, recognition và từ điển ký tự.
    std::string det_model_path = "../models/model_det.nb";
    std::string rec_model_path = "../models/model_rec.nb"; // Mô hình recognition riêng
    std::string char_dict_path = "../models/char_dict.txt";
    std::string input_dir = "../input";
    std::string output_dir = "../output";  // Ảnh có box detection được lưu lại.
    
    if (!std::filesystem::exists(input_dir)) {
        std::cerr << "Không tồn tại thư mục input: " << input_dir << std::endl;
        return -1;
    }
    std::filesystem::create_directories(output_dir);
    
    // Cấu hình cho detection.
    std::map<std::string, double> detConfig;
    detConfig["max_side_len"] = 640;        // Ảnh letterbox 640x640.
    detConfig["det_db_thresh"] = 0.8;         // Ngưỡng detection.
    detConfig["det_db_unclip_ratio"] = 1.0;
    detConfig["det_db_use_dilate"] = 1;
    
    int cpu_threads = 4;
    std::string cpu_power_mode = "LITE_POWER_HIGH";
    
    // Khởi tạo module detection và recognition.
    DetProcess detector(det_model_path, cpu_threads, cpu_power_mode);
    RecProcess recognizer(rec_model_path, char_dict_path);
    
    // Duyệt qua các ảnh trong thư mục input.
    for (const auto &entry : std::filesystem::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string file_path = entry.path().string();
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
        
        cv::Mat image = cv::imread(file_path);
        if (image.empty()) {
            std::cerr << "Không thể tải ảnh: " << file_path << std::endl;
            continue;
        }
        
        // Chạy detection.
        auto boxes = detector.detect(image, detConfig);
        
        // Vẽ các box lên ảnh gốc.
        for (const auto &box : boxes) {
            std::vector<cv::Point> pts;
            for (const auto &pt : box) {
                pts.push_back(cv::Point(pt[0], pt[1]));
            }
            if (pts.size() >= 4)
                cv::polylines(image, pts, true, cv::Scalar(0, 0, 255), 2);
        }
        
        // Lưu ảnh đã có box vào thư mục output.
        std::string output_path = output_dir + "/" + entry.path().filename().string();
        cv::imwrite(output_path, image);
        
        // Với mỗi box, crop vùng chữ và chạy recognition.
        for (const auto &box : boxes) {
            cv::Mat crop = CropBox(image, box);
            // Loại bỏ box quá nhỏ.
            if (cv::boundingRect(crop).area() < 100)
                continue;
            DecodeResult rec_result = recognizer.recognize(crop);
            std::cout << "Ảnh " << entry.path().filename().string() 
                      << " -> text: " << rec_result.text 
                      << " (confidence: " << rec_result.confidence << ")" << std::endl;
        }
    }
    
    return 0;
}
