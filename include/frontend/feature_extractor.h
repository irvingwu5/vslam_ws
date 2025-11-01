#pragma once
#include <opencv2/opencv.hpp>

class FeatureExtractor {
public:
    explicit FeatureExtractor(int n_features = 1200, int n_levels = 8, float scale_factor = 1.2f);
    void DetectAndCompute(const cv::Mat& img,
                        std::vector<cv::KeyPoint>& kps,
                        cv::Mat& desc) const;
private:
    cv::Ptr<cv::ORB> orb_;
};
