#pragma once
#include <opencv2/opencv.hpp>

struct CameraIntrinsics {
    double fx{0}, fy{0}, cx{0}, cy{0}, baseline{0};
};

class Triangulator {
public:
    explicit Triangulator(const CameraIntrinsics& Kb) : Kb_(Kb) {}
    // 根据匹配对（左像素 pl, 右像素 pr）恢复 3D 点（摄像机坐标系）
    void Triangulate(const std::vector<cv::KeyPoint>& kpsL,
                     const std::vector<cv::KeyPoint>& kpsR,
                     const std::vector<cv::DMatch>& matches,
                     std::vector<cv::Point3f>& points3d) const;

private:
    CameraIntrinsics Kb_;
};
