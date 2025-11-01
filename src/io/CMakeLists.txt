# src/io/CMakeLists.txt
add_library(io STATIC
        kitti_dataset.cpp
)

target_include_directories(io PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(io
        common
        ${OpenCV_LIBS}
)
