#include "frontend/stereo_matcher.h"
using std::vector;

StereoMatcher::StereoMatcher(const StereoMatchOptions& opt) : opt_(opt) {}

void StereoMatcher::Match(const vector<cv::KeyPoint>& kpsL, const cv::Mat& descL,
                          const vector<cv::KeyPoint>& kpsR, const cv::Mat& descR,
                          vector<cv::DMatch>& good) const {
  good.clear();
  if (descL.empty() || descR.empty()) return;

  cv::BFMatcher bf(cv::NORM_HAMMING, /*crossCheck=*/false);

  auto filter_epipolar_disp = [&](const vector<cv::DMatch>& in, vector<cv::DMatch>& out){
    out.clear();
    out.reserve(in.size());
    for (const auto& m : in) {
      const auto& pl = kpsL[m.queryIdx].pt;
      const auto& pr = kpsR[m.trainIdx].pt;
      const float dy = std::fabs(pl.y - pr.y);
      const float disp = (pl.x - pr.x);
      if (dy <= opt_.max_row_diff && disp >= opt_.min_disparity && disp <= opt_.max_disparity) {
        out.push_back(m);
      }
    }
  };

  if (opt_.ratio_test > 0.f) {
    // ratio test（可提高精确度）
    vector<vector<cv::DMatch>> knnLR, knnRL;
    bf.knnMatch(descL, descR, knnLR, 2);
    vector<cv::DMatch> candLR;
    for (auto& nn : knnLR) {
      if (nn.size() < 2) continue;
      if (nn[0].distance < opt_.ratio_test * nn[1].distance) candLR.push_back(nn[0]);
    }
    if (opt_.cross_check) {
      bf.knnMatch(descR, descL, knnRL, 2);
      vector<cv::DMatch> candRL;
      for (auto& nn : knnRL) {
        if (nn.size() < 2) continue;
        if (nn[0].distance < opt_.ratio_test * nn[1].distance) candRL.push_back(nn[0]);
      }
      // cross check
      std::vector<char> ok(descL.rows, 0);
      for (auto& m : candRL) ok[m.trainIdx] = (char)m.queryIdx + 1;
      vector<cv::DMatch> cross;
      cross.reserve(candLR.size());
      for (auto& m : candLR) {
        if (ok[m.queryIdx] && (ok[m.queryIdx]-1) == m.trainIdx) cross.push_back(m);
      }
      filter_epipolar_disp(cross, good);
    } else {
      vector<cv::DMatch> filtered;
      filter_epipolar_disp(candLR, filtered);
      good.swap(filtered);
    }
  } else {
    // 简单 match + 可选 cross check
    vector<cv::DMatch> mLR; bf.match(descL, descR, mLR);
    if (opt_.cross_check) {
      vector<cv::DMatch> mRL; bf.match(descR, descL, mRL);
      std::vector<int> fromR(descR.rows, -1);
      for (auto& m : mRL) fromR[m.queryIdx] = m.trainIdx;
      vector<cv::DMatch> cross;
      cross.reserve(mLR.size());
      for (auto& m : mLR) {
        if (fromR[m.trainIdx] == m.queryIdx) cross.push_back(m);
      }
      filter_epipolar_disp(cross, good);
    } else {
      filter_epipolar_disp(mLR, good);
    }
  }
}