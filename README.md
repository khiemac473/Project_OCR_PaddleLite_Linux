# Project_OCR_PaddleLite_Linux

📌 **Lightweight OCR System using Paddle Lite in C++**

This project implements an end-to-end Optical Character Recognition (OCR) pipeline using [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) with C++. It includes:
- Text Detection
- Text Alignment
- Text Recognition

Optimized for lightweight environments such as edge devices and embedded systems.

---

## 📁 Folder Structure

```
Project_Paddle_Lite_OCR/
├── inference_lite_lib.with_log/   # Prebuilt Paddle Lite library (v2.12)
├── model_convert/                 # Model conversion scripts/tools
│   ├── en_PP-OCRv3_det_infer/     
│   ├── en_PP-OCRv3_rec_infer/     
│   └── opt_linux                  
├── ppocr/                         # Full OCR pipeline (detect + rec)
│   ├── build/                     # Build output
│   ├── input/                     # Input images
│   ├── models/                    # Model directory
│   ├── output/                    # OCR results
│   ├── src/                       # Source code
│   └── CMakeLists.txt
├── ppocr_det/                     # Text detection + alignment only
│   ├── build/
│   ├── input/
│   ├── models/
│   ├── output/
│   ├── src/
│   └── CMakeLists.txt
├── ppocr_rec/                     # Text recognition only
│   ├── build/
│   ├── models/
│   ├── src/
│   └── CMakeLists.txt
```

---

## 🚀 Getting Started

### 🔧 Requirements
- Linux system
- Paddle Lite (prebuilt from v2.12)
- OpenCV
- CMake
- C++ compiler

> 📦 Download Paddle Lite prebuilt library from:  
> https://github.com/PaddlePaddle/Paddle-Lite/releases

Place it under `inference_lite_lib.with_log/`

---

## 🧱 Build Instructions

Example for full OCR pipeline in `ppocr/`:

```bash
cd ppocr
mkdir build && cd build
cmake ..
make
```

---

## 🧠 Acknowledgements

- [PaddlePaddle](https://github.com/PaddlePaddle)
- [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [PP-OCRv3 Models](https://github.com/PaddlePaddle/PaddleOCR)

