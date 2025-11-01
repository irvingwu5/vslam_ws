#include "frontend/triangulator.h"
using std::vector;

void Triangulator::Triangulate(const vector<cv::KeyPoint>& kpsL,
                               const vector<cv::KeyPoint>& kpsR,
                               const vector<cv::DMatch>& matches,
                               vector<cv::Point3f>& points3d) const {
    points3d.clear();
    points3d.reserve(matches.size());
    const double fx = Kb_.fx, fy = Kb_.fy, cx = Kb_.cx, cy = Kb_.cy, B = Kb_.baseline;

    for (const auto& m : matches) {
        const auto& pl = kpsL[m.queryIdx].pt;
        const auto& pr = kpsR[m.trainIdx].pt;
        const double disp = (pl.x - pr.x);
        if (disp <= 0.0) continue; // 防御
        const double Z = fx * B / disp;
        const double X = (pl.x - cx) * Z / fx;
        const double Y = (pl.y - cy) * Z / fy;
        points3d.emplace_back((float)X, (float)Y, (float)Z);
    }
}
