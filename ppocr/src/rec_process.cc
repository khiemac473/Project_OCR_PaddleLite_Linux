#include "rec_process.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

namespace ocr {

using namespace paddle::lite_api;

RecProcess::RecProcess(const std::string &model_path, const std::string &char_dict_path) {
    // Load từ điển ký tự
    char_list_ = loadCharDict(char_dict_path);
    if (char_list_.empty()) {
        std::cerr << "Không tải được từ điển ký tự từ " << char_dict_path << std::endl;
    }
    // Cấu hình Paddle Lite cho recognition
    MobileConfig config;
    config.set_model_from_file(model_path);
    predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

std::vector<std::string> RecProcess::loadCharDict(const std::string &dict_path) {
    std::vector<std::string> char_list;
    std::ifstream infile(dict_path);
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) {
            char_list.push_back(line);
        }
    }
    return char_list;
}

DecodeResult RecProcess::ctcGreedyDecoder(const float* probs, int seq_len, int num_classes, const std::vector<std::string>& char_list) {
    DecodeResult result;
    result.text = "";
    float confidence_sum = 0.f;
    int count = 0;
    int prev_index = -1;
    for (int i = 0; i < seq_len; ++i) {
        int max_index = 0;
        float max_val = probs[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            float val = probs[i * num_classes + j];
            if (val > max_val) {
                max_val = val;
                max_index = j;
            }
        }
        if (max_index != 0 && max_index != prev_index) {
            if (max_index < char_list.size()) {
                result.text += char_list[max_index];
                confidence_sum += max_val;
                count++;
            }
        }
        prev_index = max_index;
    }
    result.confidence = (count > 0) ? (confidence_sum / count) : 0.f;
    return result;
}

DecodeResult RecProcess::recognize(const cv::Mat &img) {
    // Giả sử ảnh đầu vào đã được crop chứa vùng chữ
    // Thiết lập kích thước đầu vào cố định (ví dụ: chiều cao target_h và width được tính theo tỉ lệ)
    int target_h = 48;
    int new_width = static_cast<int>(img.cols * (target_h / static_cast<float>(img.rows)));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_width, target_h));
    // Đổi sang float và chuẩn hóa [0,1]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255);
    // Nếu cần pad để có kích thước cố định (ví dụ: fixed_width), bạn có thể bổ sung ở đây.
    // Ở đây ta lấy kích thước hiện có.
    int fixed_width = new_width;
    
    std::vector<int64_t> input_shape = {1, 3, target_h, fixed_width};
    auto input_tensor = predictor_->GetInput(0);
    input_tensor->Resize(input_shape);
    float* input_data = input_tensor->mutable_data<float>();
    
    // Tách kênh (OpenCV đọc ảnh theo thứ tự BGR)
    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    size_t channel_size = target_h * fixed_width;
    // Giả sử mô hình nhận input theo thứ tự BGR (nếu cần chuyển sang RGB thì thay đổi thứ tự)
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data + c * channel_size, channels[c].data, channel_size * sizeof(float));
    }
    
    predictor_->Run();
    auto output_tensor = predictor_->GetOutput(0);
    auto output_shape = output_tensor->shape(); // [1, seq_len, num_classes]
    int seq_len = output_shape[1];
    int num_classes = output_shape[2];
    const float* output_data = output_tensor->data<float>();
    
    return ctcGreedyDecoder(output_data, seq_len, num_classes, char_list_);
}

} // namespace ocr
