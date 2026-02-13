cmake_minimum_required(VERSION 3.16)
project(Simple_Recorder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(BLADERF REQUIRED libbladeRF)

add_executable(Simple_Recorder
    src/main.cpp
)

target_include_directories(Simple_Recorder PRIVATE ${BLADERF_INCLUDE_DIRS})
target_link_libraries(Simple_Recorder PRIVATE ${BLADERF_LIBRARIES})
