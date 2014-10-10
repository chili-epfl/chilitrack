cmake_minimum_required(VERSION 2.8)

project(chilitags)

add_definitions(-std=c++11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(tracker src/tracker.cpp)
target_link_libraries(tracker ${OpenCV_LIBRARIES})

