#include <iostream>
#include <fstream>
#include <opencv2/highgui.hpp>

#include "frontend/feature_extractor.h"
#include "frontend/stereo_matcher.h"
#include "frontend/triangulator.h"

// 你已有的头：数据集+标定
#include "io/kitti_dataset.h"   // 假设你有
#include "common/config.h"      // 假设你有

int main(int argc, char** argv) {
  std::string config = (argc > 1) ? argv[1] : "../config/kittistereo.yaml";

  // 数据集：请确保有接口能提供该帧的 fx,fy,cx,cy,baseline
  KittiDataset dataset(config);

  FeatureExtractor extractor(1200, 8, 1.2f);
  StereoMatcher matcher({1.5f, 1.f, 160.f, true, 0.8f});

  int frame_id = 0;
  cv::Mat left, right;

  // —— 示例：如果你的 Dataset 能返回 StereoCamera ——
  // struct StereoCamera { cv::Mat K_left; double baseline; ... };
  StereoCamera cam;

  auto LoadStereoAt = [&](const std::string& config, int frame_id, cv::Mat& left, cv::Mat& right, StereoCamera& cam) -> bool{
        if(frame_id < 0) return false;
        KittiDataset ds(config);
        for(int i=0; i<=frame_id; ++i){
            if(!ds.NextStereo(left, right, cam)) return false;
        }
        return true;
  };

  // 先加载第 0 帧
  if (!LoadStereoAt(config, frame_id, left, right, cam)) {
      std::cerr << "Failed to load first frame." << std::endl;
      return -1;
  }
  while (true) {
    // 如果你当前的 NextStereo 只能返回图像，请在 KittiDataset 中新增一个
    // GetCurrentIntrinsics() 或 NextStereo(left,right,cam) 的重载。
    // 这里展示两种写法之一（首选重载）：

    // ① 推荐（同时取图像与本帧标定）
    //if (!dataset.NextStereo(left, right, cam)) break;

    // ② 临时（图像与标定分步获取）
    CameraIntrinsics Kb{};
    // TODO: 从你的 Dataset / StereoCamera 中取 fx,fy,cx,cy,baseline 填入：
    // Kb.fx = ..., Kb.fy = ..., Kb.cx = ..., Kb.cy = ..., Kb.baseline = ...
    // 例如：
    Kb.fx = cam.K_left.at<double>(0,0);
    Kb.fy = cam.K_left.at<double>(1,1);
    Kb.cx = cam.K_left.at<double>(0,2);
    Kb.cy = cam.K_left.at<double>(1,2);
    Kb.baseline = cam.baseline;

    // —— 特征提取 ——
    std::vector<cv::KeyPoint> kpsL, kpsR; cv::Mat dL, dR;
    extractor.DetectAndCompute(left, kpsL, dL);
    extractor.DetectAndCompute(right, kpsR, dR);

    // —— 匹配 ——
    std::vector<cv::DMatch> good;
    matcher.Match(kpsL, dL, kpsR, dR, good);


    // —— 可视化匹配 ——
    cv::Mat vis;
    cv::drawMatches(left, kpsL, right, kpsR, good, vis);
    cv::putText(vis, "matches: " + std::to_string(good.size()),
                {20,30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, 255, 2);
    cv::imshow("Stereo Matches", vis);
    //if ((cv::waitKey(1) & 0xff) == 27) break;

    // —— 三角化 ——
    Triangulator tri(Kb);
    std::vector<cv::Point3f> pts3d;
    tri.Triangulate(kpsL, kpsR, good, pts3d);

    // —— 简单统计 ——
    if (!pts3d.empty()) {
      float zmin=1e9f, zmax=0.f, zmed=0.f;
      std::vector<float> zs; zs.reserve(pts3d.size());
      for (auto& p: pts3d) { zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z); zs.push_back(p.z); }
      std::nth_element(zs.begin(), zs.begin()+zs.size()/2, zs.end());
      zmed = zs[zs.size()/2];
      std::cout << "Frame " << frame_id << " 3D points: " << pts3d.size()
                << "  Z[min/med/max]= " << zmin << " / " << zmed << " / " << zmax << " m\n";
    }
      int key = cv::waitKey(0) & 0xff;
    if(key == 27 || key == 'q' || key == 'Q') break; //esc or q
    if(key == 'n' || key == 'N'){
       int next_id = frame_id + 1;
       if(LoadStereoAt(config, next_id, left, right, cam)){
           frame_id = next_id;
       }else {
           std::cout << "Already at last frame." << std::endl;
       }
    }else if (key == 'b' || key == 'B') {
        if (frame_id > 0) {
            int prev_id = frame_id - 1;
            if (LoadStereoAt(config, prev_id, left, right, cam)) {
                frame_id = prev_id;
            }
        }else {
            std::cout << "Already at first frame." << std::endl;
        }
    }
  }
  return 0;
}