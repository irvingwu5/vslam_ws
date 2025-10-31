#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <string>
//结构体成员默认public，纯数据载体，直接暴露成员变量供外部读写
struct StereoCamera{
    cv::Mat K_left, K_right; //左右相机内参
    cv::Mat D_left, D_right; //左右相机畸变系数
    cv::Mat R, T; //左右相机之间的旋转平移
    double baseline; //双目相机基线
};
//class成员默认private
class KittiDataset{
public:
    //使用config对象读文件夹路径
    explicit KittiDataset(const std::string& config_file); //构造函数
    //加载实际数据对
    bool NextStereo(cv::Mat& left, cv::Mat& right, StereoCamera& cam);
private:
    bool LoadFilesPaths();//加载图像对和对应标定文件绝对路径
    //加载实际标定数据到双目相机模型结构体
    bool LoadCalibration(const std::string& calib_file, StereoCamera& cam);

    std::vector<std::string> left_imgs_, right_imgs_; //存放每张图片的绝对路径
    std::vector<std::string> calib_files_; //存放每个标定文件的绝对路径
    size_t current_index_ = 0; //读取第几组数据对
    std::string left_dir_, right_dir_, calib_dir_; //文件夹路径
};