# apps/CMakeLists.txt

# 自动搜索 apps 下的所有 cpp 文件
file(GLOB APP_SOURCES *.cpp)

foreach(app_src ${APP_SOURCES})
    get_filename_component(app_name ${app_src} NAME_WE)
    add_executable(${app_name} ${app_src})

    target_link_libraries(${app_name}
            frontend
            io
            common
            ${OpenCV_LIBS}
    )

    message(STATUS "Created executable: ${app_name}")
endforeach()
