#include <io/kitti_dataset.h>
#include <opencv2/highgui.hpp>
int main() {
    //解析配置
    std::string config_file = "/home/wxy/Documents/CppProjects/vslam_ws/config/kittistereo.yaml";
    //初始化数据集
    KittiDataset dataset(config_file);//path -> config -> kitti -> private var
    //初始化左右图像、相机、当前帧下标
    cv::Mat left, right;
    StereoCamera cam;
    int frame_id = 0;
    //读取数据帧
    while (dataset.NextStereo(left,right,cam)) {
        //显示加载的数据
        std::cout << "Frame: " << frame_id++
                  << " | Baseline: " << cam.baseline << "m" << std::endl;
        cv::Mat stereo;
        cv::hconcat(left, right, stereo);
        cv::imshow("Stereo", stereo);
        if (cv::waitKey(10) == 27) break;
    }
    return 0;
}
