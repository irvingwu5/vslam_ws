#pragma once
#include <opencv2/opencv.hpp>

struct StereoMatchOptions {
    float max_row_diff = 1.5f;   // 行约束：校正后应同一行
    float min_disparity = 1.f;   // 视差下限
    float max_disparity = 160.f; // 视差上限
    bool cross_check = true;     // 交叉验证
    float ratio_test = 0.f;      // 可选：0=不用，0.8~0.9 常用
};

class StereoMatcher {
public:
    explicit StereoMatcher(const StereoMatchOptions& opt = {});
    // 输入左右关键点与描述子，输出过滤后的匹配
    void Match(const std::vector<cv::KeyPoint>& kpsL, const cv::Mat& descL,
               const std::vector<cv::KeyPoint>& kpsR, const cv::Mat& descR,
               std::vector<cv::DMatch>& good) const;
private:
    StereoMatchOptions opt_;
};
