
file(GLOB_RECURSE sources *.cpp)

add_library(
        chilitrack
        SHARED
        ${sources}
)

target_link_libraries(chilitrack ${OpenCV_LIBS})

install (TARGETS chilitrack LIBRARY DESTINATION lib)

file(GLOB_RECURSE chilitrack_headers ../include/*)

install(FILES
    ${chilitrack_headers}
    DESTINATION include/chilitrack
    )

