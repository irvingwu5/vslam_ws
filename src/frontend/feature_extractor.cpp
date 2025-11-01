#include<frontend/feature_extractor.h>

FeatureExtractor::FeatureExtractor(int n_features, int n_levels, float scale_factor) {
    orb_ = cv::ORB::create(n_features, scale_factor, n_levels);
}

void FeatureExtractor::DetectAndCompute(const cv::Mat &img, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) const {
    orb_ ->detectAndCompute(img, cv::noArray(), kps, desc);
}