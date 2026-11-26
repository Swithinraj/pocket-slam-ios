cmake_minimum_required(VERSION 3.16)
project(slamcore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(slamcore
  src/preintegration.cpp
  src/tracker.cpp
  src/estimator.cpp
  src/system.cpp
  src/slamcore_c.cpp)
target_include_directories(slamcore PUBLIC include src/public third_party/eigen PRIVATE src)
target_compile_options(slamcore PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(run_recording tools/run_recording.cpp)
target_include_directories(run_recording PRIVATE third_party/stb)
target_link_libraries(run_recording PRIVATE slamcore)

add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE slamcore)
