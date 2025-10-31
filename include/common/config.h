//关于读取yaml文件的头文件
//头文件作用：进行类的定义，包括数据成员和函数成员的声明
#pragma once
#include<string>
#include <opencv2/core.hpp>

class Config{
public:
    //构造函数，用于在创建对象时自动初始化对象的状态
    Config() = default; //默认构造函数(无参数)
    //explicit防止隐式转换 不允许Config cfg = "config.txt" 仅允许 Config cfg("config.txt")
    //如果构造函数没有 explicit，这行代码会隐式将字符串转换为 Config 对象
    //参数为：传入的配置文件路径
    explicit Config(const std::string& filename); //带参数显式构造函数
    //成员函数声明,const成员函数,在该函数中不允许修改成员变量
    std::string GetLeftImageDir() const { return left_gray_dir_; };
    std::string GetRightImageDir() const { return right_gray_dir_; };
    std::string GetCalibDir() const { return calib_dir_; };
//成员变量声明
private:
    std::string left_gray_dir_;
    std::string right_gray_dir_;
    std::string calib_dir_;
};

