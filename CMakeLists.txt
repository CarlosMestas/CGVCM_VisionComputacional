cmake_minimum_required(VERSION 3.5)

project(OpenCV_Example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCV_Example Image.cpp Image.h RGB.cpp RGB.h main.cpp)
set_target_properties(
    OpenCV_Example
    PROPERTIES
        OUTPUT_NAME "executable"
        SUFFIX ""
)


target_link_libraries(OpenCV_Example ${OpenCV_LIBS})
