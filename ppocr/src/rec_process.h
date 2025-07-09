#pragma once
#include <string>
#include <vector>
#include "opencv2/core.hpp"
#include "paddle_api.h"

namespace ocr {

struct DecodeResult {
    std::string text;
    float confidence;
};

class RecProcess {
public:
    // Khởi tạo với đường dẫn mô hình recognition và từ điển ký tự.
    RecProcess(const std::string &model_path, const std::string &char_dict_path);
    
    // Hàm recognize: nhận ảnh (crop vùng chữ) và trả về kết quả decode.
    DecodeResult recognize(const cv::Mat &img);

private:
    // Hàm load từ điển ký tự từ file.
    std::vector<std::string> loadCharDict(const std::string &dict_path);
    // Hàm giải mã CTC theo phương pháp greedy.
    DecodeResult ctcGreedyDecoder(const float* probs, int seq_len, int num_classes, const std::vector<std::string>& char_list);

    std::vector<std::string> char_list_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};

} // namespace ocr
