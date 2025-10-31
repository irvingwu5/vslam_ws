cmake_minimum_required(VERSION 3.10)
project(vslam)
# 设置变量值
set(CMAKE_CXX_STANDARD 17) # c++标准为17
set(CMAKE_BUILD_TYPE DEBUG) # 构建类型
# 查找 OpenCV 库，REQUIRED 表示必须找到，否则报错
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
# 头文件搜索路径
# 告诉编译器 “去哪里找头文件”，后续代码中 #include <xxx.h> 会搜索这些路径
include_directories(include
        ${OpenCV_INCLUDE_DIRS})
# 生成名为 vslam 的静态库（默认静态库，加 SHARED 可生成动态库）
add_library(vslam
        src/common/config.cpp
        src/io/kitti_dataset.cpp)
# 给 vslam 库链接 OpenCV 依赖
target_link_libraries(vslam PUBLIC ${OpenCV_LIBS} yaml-cpp)

# 生成名为 run_kitti 的可执行文件
add_executable(run_kitti apps/run_kitti.cpp)
# 关键修复：链接到 vslam 和 OpenCV
target_link_libraries(run_kitti PRIVATE vslam)