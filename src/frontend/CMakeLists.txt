# src/frontend/CMakeLists.txt
add_library(frontend STATIC
        feature_extractor.cpp
        stereo_matcher.cpp
        triangulator.cpp
)

target_include_directories(frontend PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(frontend
        io
        common
        ${OpenCV_LIBS}
)
