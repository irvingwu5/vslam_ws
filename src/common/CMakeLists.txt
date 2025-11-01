# src/common/CMakeLists.txt
# 模块编译为静态库common
add_library(common STATIC
        config.cpp
)
# 给目标common添加头文件搜索路径（编译期）
target_include_directories(common PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)
# 给目标common添加要链接的库（链接期）
target_link_libraries(common
        PUBLIC yaml-cpp
        PRIVATE ${OpenCV_LIBS}
)
