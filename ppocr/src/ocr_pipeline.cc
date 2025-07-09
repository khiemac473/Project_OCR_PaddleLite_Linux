#include <iostream>
#include <filesystem>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "opencv2/imgcodecs.hpp"
#include "det_process.h"
#include "rec_process.h"

namespace fs = std::filesystem;
using namespace ocr;

int main() {
    // Đường dẫn mô hình detection và recognition, từ điển ký tự.
    std::string det_model_path = "../models/model_det.nb";
    std::string rec_model_path = "../models/model_rec.nb";  // Giả sử có mô hình recognition riêng
    std::string char_dict_path = "../models/char_dict.txt";
    
    std::string input_dir = "../input";
    std::string det_output_dir = "../output";  // Ảnh sau detection (vẽ box)
    
    if (!fs::exists(input_dir)) {
        std::cerr << "Không tồn tại thư mục input: " << input_dir << std::endl;
        return -1;
    }
    fs::create_directories(det_output_dir);
    
    // Khởi tạo module detection và recognition.
    int cpu_threads = 4;
    std::string cpu_power_mode = "LITE_POWER_HIGH";
    DetProcess detector(det_model_path, cpu_threads, cpu_power_mode);
    RecProcess recognizer(rec_model_path, char_dict_path);
    
    // Cấu hình cho detection
    std::map<std::string, double> detConfig;
    detConfig["max_side_len"] = 640;        // Ảnh letterbox 640x640
    detConfig["det_db_thresh"] = 0.8;         // Ngưỡng detection
    detConfig["det_db_unclip_ratio"] = 1.5;  
    detConfig["det_db_use_dilate"] = 0;
    
    // Duyệt qua từng ảnh trong thư mục input.
    for (const auto &entry : fs::directory_iterator(input_dir)) {
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
        
        // Vẽ box lên ảnh gốc.
        for (const auto &box : boxes) {
            std::vector<cv::Point> pts;
            for (const auto &pt : box) {
                pts.push_back(cv::Point(pt[0], pt[1]));
            }
            if (pts.size() >= 4)
                cv::polylines(image, pts, true, cv::Scalar(0, 0, 255), 2);
        }
        
        // Lưu ảnh detection.
        std::string output_path = det_output_dir + "/" + entry.path().filename().string();
        cv::imwrite(output_path, image);
        
        // Với mỗi box hợp lệ, crop vùng chữ và chạy recognition.
        for (const auto &box : boxes) {
            cv::Mat crop = CropBox(image, box);
            // Có thể bổ sung bộ lọc dựa trên diện tích (ví dụ: bỏ box nhỏ)
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
