project(mathematical_robotics CXX)

cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 20) 

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Matplot++ REQUIRED)

include_directories(${CERES_INCLUDE_DIR})
include_directories(../math_utils)

set(libs 
    Eigen3::Eigen
    Ceres::ceres
    Matplot++::matplot
)

add_executable(point_cloud_matching "point_cloud_matching.cpp")
target_link_libraries(point_cloud_matching 
    ${libs}
)