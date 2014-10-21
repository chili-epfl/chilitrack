cmake_minimum_required(VERSION 2.8)

project(chilitrack)

set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")


add_definitions(-std=c++11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV COMPONENTS video highgui videoio features2d calib3d REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DCHILITRACK_DEBUG)
endif()

include_directories(include)

add_subdirectory(src)

add_executable(tracker samples/tracker.cpp)
target_link_libraries(tracker chilitrack ${OpenCV_LIBRARIES})

add_executable(minimal samples/minimal.cpp)
target_link_libraries(minimal chilitrack ${OpenCV_LIBRARIES})

##########################################
##       PKGCONFIG file                ###
##########################################

if(EXISTS "${CMAKE_SOURCE_DIR}/chilitrack.pc.in")
    message(STATUS "configuring ${CMAKE_SOURCE_DIR}/chilitrack.pc.in --> ${CMAKE_CURRENT_BINARY_DIR}/chilitrack.pc")

    # build a list of -D<...> macros to store in pkgconfig CFlags
    get_directory_property(INTERNAL_COMPILE_FLAGS COMPILE_DEFINITIONS)
    foreach(DEF ${INTERNAL_COMPILE_FLAGS})
        set(COMPILE_FLAGS "${COMPILE_FLAGS} -D${DEF}")
    endforeach()

    configure_file(${CMAKE_SOURCE_DIR}/chilitrack.pc.in 
      ${CMAKE_CURRENT_BINARY_DIR}/chilitrack.pc
      @ONLY )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/chilitrack.pc
        DESTINATION lib/pkgconfig
    )
else()
    message(STATUS "${CMAKE_SOURCE_DIR}/chilitrack.pc.in not found!")

endif()


