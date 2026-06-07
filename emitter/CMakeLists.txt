cmake_minimum_required(VERSION 3.15)
project(emitter)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake")

find_package(CURL REQUIRED)

add_executable(emitter src/main.cpp)
target_link_libraries(emitter CURL::libcurl)