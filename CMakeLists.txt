cmake_minimum_required(VERSION 2.8)

project(chilitags)

add_definitions(-std=c++11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV COMPONENTS video highgui videoio features2d calib3d REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DCHILITRACK_DEBUG)
endif()

include_directories(include)
add_executable(tracker samples/tracker.cpp src/tracker.cpp src/template.cpp)
target_link_libraries(tracker ${OpenCV_LIBRARIES})

