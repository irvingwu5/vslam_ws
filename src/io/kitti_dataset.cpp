#include "io/kitti_dataset.h"
#include "common/config.h"
#include <fstream>
#include <sstream>
//构造函数实现
KittiDataset::KittiDataset(const std::string& config_file){
    Config cfg(config_file);
    left_dir_ = cfg.GetLeftImageDir();//读取左相机目录路径
    right_dir_ = cfg.GetRightImageDir();//读取右相机目录路径
    calib_dir_ = cfg.GetCalibDir();//读取标定文件目录路径
    //检查加载是否完成
    if(!LoadFilesPaths()) {
        std::cerr << "[KittiDataset] Failed to load image paths!" << std::endl;
    }
    //为什么构造函数中不加载标定文件？
    //标定文件是在 每帧读取时 再动态加载，而不是在初始化时统一加载
    //构造函数通常只负责初始化静态资源（例如目录、文件列表）
    //标定文件是与帧同步的动态资源，必须在读取图像时匹配加载
}
//成员函数LoadImagePaths加载图像路径而非实际文件，枚举左右图像文件
bool KittiDataset::LoadFilesPaths() {
    namespace fs = std::filesystem;
    //检查左右图像和标定文件目录是否存在
    if (!fs::exists(left_dir_) || !fs::exists(right_dir_) || !fs::exists(calib_dir_)) {
        std::cerr << "[KittiDataset] Invalid directory!" << std::endl;
        return false;
    }
    //循环遍历每张左图片并将每张图片完整路径存储到vector中
    for (const auto& entry : fs::directory_iterator(left_dir_))
        left_imgs_.push_back(entry.path().string());
    //循环遍历每张右图片并将每张图片完整路径存储到vector中
    for(const auto& entry : fs::directory_iterator(right_dir_))
        right_imgs_.push_back(entry.path().string());
    //循环遍历每个标定文件并将每个标定文件完整路径存储到vector中
    for (const auto& entry : fs::directory_iterator(calib_dir_))
        calib_files_.push_back(entry.path().string());
    //为了能够顺序读取，对vector进行排序
    std::sort(left_imgs_.begin(), left_imgs_.end());
    std::sort(right_imgs_.begin(),right_imgs_.end());
    std::sort(calib_files_.begin(),calib_files_.end());
    //为了保证左右图片和标定文件一对一，只保留最小数量的文件
    size_t n =std::min({left_imgs_.size(),right_imgs_.size(),calib_files_.size()});
    left_imgs_.resize(n);
    right_imgs_.resize(n);
    calib_files_.resize(n);
    //数据集所有文件的路径收集完成
    std::cout << "[KittiDataset] Loaded " << n << "stereo pairs with calibration." << std::endl;
    return true;
}

//成员函数LoadCalibration加载标定文件，读取并解析 calib 文件
bool KittiDataset::LoadCalibration(const std::string &calib_file, StereoCamera &cam) {
    //打开帧对应的标定文件
    std::ifstream fin(calib_file);
    //检查是否打开成功
    if (!fin.is_open()) {
        std::cerr << "[KittiDataset] Cannot open calib file: " << calib_file << std::endl;
        return false;
    }
    //解析标定文件中灰度相机相关行的标定参数
    std::string line;
    //P0左灰度相机投影矩阵、P1右灰度相机投影矩阵
    cv::Mat P0(3,4,CV_64F), P1(3,4,CV_64F);
    while (std::getline(fin,line)) {
        //找文件中行首是P0
        if (line.rfind("P0:",0) == 0) {
            std::stringstream ss(line.substr(3));
            //3*4=12个数分别填充到P0矩阵中
            //整除结果每 4 个数递增 1（0~3 为 0，4~7 为 1，8~11 为 2）
            //取模结果以 4 为周期循环（0,1,2,3,0,1,2,3,...）
            for (int i=0; i<12; ++i) ss >> P0.at<double>(i/4, i%4);
        }else if (line.rfind("P1:",0) == 0) {
            std::stringstream ss(line.substr(3));
            for (int i=0; i<12; ++i) ss >> P1.at<double>(i/4, i%4);
        }
    }
    fin.close();
    //将参数并赋值到相机结构体中对应参数
    //P=K[R|t],双目相机只有平移无旋转
    //P0=K[I|0] P1=K[I|t]
    cam.K_left = P0(cv::Rect(0,0,3,3)).clone();
    cam.K_right = P1(cv::Rect(0,0,3,3)).clone();
    //x_distorted=x(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy+p_2(r^2+2x^2)
    //y_distorted=y(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy
    //5个标定参数k_1、k_2、k_3、p_1、p_2
    cam.D_left = cv::Mat::zeros(1,5,CV_64F);
    cam.D_right = cv::Mat::zeros(1,5,CV_64F);
    //R为单位矩阵、T为P1的平移向量tx,ty=0,tz=0值
    cam.R = cv::Mat::eye(3,3,CV_64F);
    //P0=K[I|0] P1=K[I|t],t中数值=[fx*tx,0,0]，将tx从投影矩阵P1中分解出来
    //tx=(t中第一个数值结果)/fx，即表示两个灰度相机之间的水平距离
    //第一个灰度相机cam0 z方向朝前、y方向朝下、x方向朝第二个灰度相机cam1
    //cam1相机坐标系原点在cam0坐标系中(世界坐标系)的位置
    cam.T = (cv::Mat_<double>(3,1) << -P1.at<double>(0,3)/P1.at<double>(0,0),0,0);
    //baseline B=|t_x|
    cam.baseline = std::abs(cam.T.at<double>(0));
    //返回加载状态
    return true;
}

//成员函数NextStereo加载下一组左右图像，按序读取灰度图像
bool KittiDataset::NextStereo(cv::Mat &left, cv::Mat &right, StereoCamera &cam) {
    //下标判断遍历到第几组
    if (current_index_ >= left_imgs_.size()) return false;
    //读取左图像
    left = cv::imread(left_imgs_[current_index_], cv::IMREAD_GRAYSCALE);
    //读取右图像
    right = cv::imread(right_imgs_[current_index_], cv::IMREAD_GRAYSCALE);
    //检查当前图像是否为空
    if (left.empty() || right.empty()) return false;
    //当前对应标定文件路径
    std::string calib_file = calib_files_[current_index_];
    //加载对应标定文件
    if (!LoadCalibration(calib_file, cam)) {
        std::cerr << "[KittiDataset] Failed to load calib for frame " << std::endl;
        return false;
    }
    //下标加一读取下一组
    ++current_index_;
    //返回状态
    return true;
}
