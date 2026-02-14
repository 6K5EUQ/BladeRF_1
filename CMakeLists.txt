cmake_minimum_required(VERSION 3.16)
project(FFT_Analyzer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(BLADERF REQUIRED libbladeRF)
pkg_check_modules(FFTW REQUIRED fftw3f)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

set(IMGUI_PATH /usr/include/imgui)
set(IMGUI_BACKENDS_PATH ${IMGUI_PATH}/backends)

set(IMGUI_SOURCES
    ${IMGUI_PATH}/imgui.cpp
    ${IMGUI_PATH}/imgui_demo.cpp
    ${IMGUI_PATH}/imgui_draw.cpp
    ${IMGUI_PATH}/imgui_tables.cpp
    ${IMGUI_PATH}/imgui_widgets.cpp
    ${IMGUI_BACKENDS_PATH}/imgui_impl_glfw.cpp
    ${IMGUI_BACKENDS_PATH}/imgui_impl_opengl3.cpp
)

add_executable(fft_analyzer
    src/main.cpp
    src/streaming_viewer.cpp
    src/record_with_fft.cpp
    ${IMGUI_SOURCES}
)

target_include_directories(fft_analyzer PRIVATE
    ${BLADERF_INCLUDE_DIRS}
    ${FFTW_INCLUDE_DIRS}
    ${IMGUI_PATH}
    ${IMGUI_BACKENDS_PATH}
)

target_link_libraries(fft_analyzer PRIVATE
    ${BLADERF_LIBRARIES}
    ${FFTW_LIBRARIES}
    OpenGL::GL
    GLEW::GLEW
    glfw
    pthread
)

target_compile_options(fft_analyzer PRIVATE -O3 -march=native)

add_executable(view_fft src/view_fft.cpp ${IMGUI_SOURCES})

target_include_directories(view_fft PRIVATE 
    ${IMGUI_PATH}
    ${IMGUI_BACKENDS_PATH}
    ${FFTW_INCLUDE_DIRS}
)

target_link_libraries(view_fft PRIVATE 
    OpenGL::GL 
    GLEW::GLEW 
    glfw
    ${FFTW_LIBRARIES}
)

target_compile_options(view_fft PRIVATE -O3 -march=native)