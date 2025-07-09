#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "paddle_api.h"

using namespace paddle::lite_api;

//decode 
struct DecodeResult {
    std::string text;
    float confidence;
};
// load char_dict.txt
std::vector<std::string> load_char_dict(const std::string& dict_path) {
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
// Hàm giải mã CTC
DecodeResult ctc_greedy_decoder(const float* probs, int seq_len, int num_classes, const std::vector<std::string>& char_list) {
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

int main() {
    std::string model_dir = "../models/";
    std::string model_file = model_dir + "model.nb";
    std::string char_dict_path = model_dir + "char_dict.txt";
    auto char_list = load_char_dict(char_dict_path);
    if (char_list.empty()) {
        std::cerr << "Không tải được từ điển ký tự từ " << char_dict_path << std::endl;
        return -1;
    }
    // Cấu hình Paddle Lite predictor
    MobileConfig config;
    config.set_model_from_file(model_file);
    auto predictor = CreatePaddlePredictor<MobileConfig>(config);
    // Thư mục chứa ảnh đầu ra từ detection
    std::string image_dir = "../../ppocr_det/output/";
    std::vector<cv::String> image_files;
    cv::glob(image_dir + "*.jpg", image_files, false);

    if (image_files.empty()) {
        std::cerr << "Không tìm thấy ảnh trong thư mục: " << image_dir << std::endl;
        return -1;
    }
    // Cấu hình kích thước cố định: chiều cao target_h và fixed_width
    int target_h = 48;
    // int fixed_width = 195;
    
    for (const auto& img_file : image_files) {
        cv::Mat img = cv::imread(img_file);
        if (img.empty()) {
            std::cerr << "Không đọc được ảnh: " << img_file << std::endl;
            continue;
        }
        int fixed_width = (int)((float) target_h / img.rows * img.cols);
        // int fixed_width = 195;

        printf("%d %d\n", img.cols, img.rows);
        printf("%d %d\n", fixed_width, target_h);
        // Resize ảnh về chiều cao target_h, giữ tỉ lệ
        int new_width = static_cast<int>(img.cols * (target_h / static_cast<float>(img.rows)));
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_width, target_h));
        // Chuyển đổi sang float và chuẩn hóa
        resized.convertTo(resized, CV_32FC3, 1.0 / 255);
        // Nếu new_width < fixed_width thì pad, nếu lớn hơn thì resize lại về fixed_width
        cv::Mat padded;
        if (new_width < fixed_width) {
            padded = cv::Mat::zeros(target_h, fixed_width, CV_32FC3);
            // Copy ảnh vào góc trái của ảnh padded
            resized.copyTo(padded(cv::Rect(0, 0, new_width, target_h)));
        } else {
            // Nếu ảnh sau resize có chiều rộng vượt fixed_width, ta resize lại về fixed_width
            cv::resize(resized, padded, cv::Size(fixed_width, target_h));
        }
        // Tách các kênh (ở đây OpenCV đọc ảnh theo thứ tự BGR)
        std::vector<cv::Mat> channels;
        cv::split(padded, channels);
        if (channels.size() != 3) {
            std::cerr << "Số lượng kênh không đúng cho ảnh: " << img_file << std::endl;
            continue;
        }
        // Chuẩn bị tensor input với shape [1, 3, target_h, fixed_width]
        std::vector<int64_t> input_shape = {1, 3, target_h, fixed_width};
        auto input_tensor = predictor->GetInput(0);
        input_tensor->Resize(input_shape);
        float* input_data = input_tensor->mutable_data<float>();
        size_t channel_size = target_h * fixed_width;
        // Nếu mô hình cần RGB, đổi thứ tự kênh (ví dụ: channels[2-c]) nếu cần
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_data + c * channel_size, channels[c].data, channel_size * sizeof(float));
        }
        // Chạy inference
        predictor->Run();
        // Lấy output: giả sử output có shape [1, seq_len, 97]
        auto output_tensor = predictor->GetOutput(0);
        auto output_shape = output_tensor->shape();
        int seq_len = output_shape[1];
        int num_classes = output_shape[2];
        const float* output_data = output_tensor->data<float>();
        // Giải mã kết quả và lấy thông tin độ tin cậy
        DecodeResult decode_result = ctc_greedy_decoder(output_data, seq_len, num_classes, char_list);
        std::cout << "Image: " << img_file << " -> " << decode_result.text 
                  << " (confidence: " << decode_result.confidence << ")" << std::endl;
    }
    return 0;
}
