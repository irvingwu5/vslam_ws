cmake_minimum_required(VERSION 3.10)
project(vslam_ws LANGUAGES CXX)

# --- 基础设置 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- 查找依赖 ---
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- 全局 include 目录 ---
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# --- 子模块 ---
add_subdirectory(src/common)
add_subdirectory(src/io)
add_subdirectory(src/frontend)
add_subdirectory(apps)

# --- 第三方依赖（可选）---
if(EXISTS "${PROJECT_SOURCE_DIR}/thirdparty/CMakeLists.txt")
    add_subdirectory(thirdparty)
endif()
