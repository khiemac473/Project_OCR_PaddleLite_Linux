cmake_minimum_required(VERSION 3.10)
project(ocr_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -pthread")

# Đường dẫn đến Paddle Lite và MKLML
set(PADDLE_LITE_DIR "${PROJECT_SOURCE_DIR}/../inference_lite_lib.with_log/cxx")
set(MKLML_DIR "${PROJECT_SOURCE_DIR}/../inference_lite_lib.with_log/third_party/mklml")

# Include Paddle Lite và MKLML
include_directories(${PADDLE_LITE_DIR}/include ${MKLML_DIR}/include)
link_directories(${PADDLE_LITE_DIR}/lib ${MKLML_DIR}/lib)

# Thêm thư mục src chứa code nguồn
include_directories(${PROJECT_SOURCE_DIR}/src)

# Tìm và cấu hình OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "Không tìm thấy OpenCV!")
endif()

# Thêm executable với các file nguồn trong src
add_executable(ocr_detect
    src/main.cc
    src/clipper.cpp
    src/db_post_process.cc
)

# Liên kết thư viện
if(WIN32)
    target_link_libraries(ocr_detect libpaddle_api_full_bundled.lib shlwapi.lib)
else()
    target_link_libraries(ocr_detect -lpaddle_full_api_shared -liomp5 -ldl ${OpenCV_LIBS})
endif()

