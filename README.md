# mathematical_robotics

[![build](https://github.com/TakumaNakao/mathematical_robotics/actions/workflows/build.yml/badge.svg)](https://github.com/TakumaNakao/mathematical_robotics/actions/workflows/build.yml)

## What is this?
Demonstration of algorithms for robotics.  

## Dependencies
* CMake
* Eigen3
* Ceres Solver
* Matplot++

## Docker
Launch Docker with the following command:
```
bash docker_bringup.sh
```

## Build
build
```
bash build.sh
```
clean build
```
bash clean_build.sh
```

## Point Cloud Matching
Optimization of point cloud using Lie algebra.  
Using self created Gauss-Newton method and Ceres Solver for optimization.

### point_cloud_matching
Pure matching between 3D point clouds.

#### 例
![point_cloud_matching](readme_img/point_cloud_matching.gif)

### point_cloud_to_line_matching
Matching a point cloud with a line segment on a 2D plane.

#### 例
![point_cloud_to_line_matching](readme_img/point_cloud_to_line_matching.gif)

## References
* https://qiita.com/scomup/items/fa9aed8870585e865117
* https://github.com/scomup/MathematicalRobotics/tree/main
* https://github.com/borglab/gtsam/tree/develop