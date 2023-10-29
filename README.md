# mathematical_robotics

[![build](https://github.com/TakumaNakao/mathematical_robotics/actions/workflows/build.yml/badge.svg)](https://github.com/TakumaNakao/mathematical_robotics/actions/workflows/build.yml)

## 環境
* CMake
* Eigen3
* Ceres Solver
* Matplot++

## Docker
以下のコマンドでDockerを起動できる
```
bash docker_bringup.sh
```

## point_cloud_matching
リー群による剛体変換を用いた3次元点群のマッチング  
最適化にはCeres Solverを使用  

### 例
<img src="img/point_cloud_matching_before.png" width="50%"><img src="img/point_cloud_matching_after.png" width="50%">
