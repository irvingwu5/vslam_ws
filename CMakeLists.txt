cmake_minimum_required(VERSION 3.10)
project(vslam)
# 设置变量值
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE DEBUG)
# 头文件搜索路径
include_directories(include)
# 指定要生成的可执行文件和其源文件
add_executable(vslam src/main.cpp)