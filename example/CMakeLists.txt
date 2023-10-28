project(mathematical_robotics CXX)

cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 20) 

include(GNUInstallDirs)

find_package(Python3 REQUIRED)
find_package(Python3 COMPONENTS Development.Embed REQUIRED)
find_package(Python3 COMPONENTS NumPy REQURIED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIR})
include_directories(../math_utils)
include_directories(../matplotlibcpp)

set(libs 
    Python3::Python
    Python3::Module
    Python3::NumPy
    Eigen3::Eigen
    Ceres::ceres
)

add_executable(point_cloud_matching "point_cloud_matching.cpp")
target_link_libraries(point_cloud_matching 
    ${libs}
)