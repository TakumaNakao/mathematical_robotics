#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <string>
#include <random>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "math_utils.hpp"
#include "lie_group.hpp"
#include "matplotlibcpp.hpp"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

struct CostFunctor {
    template<typename T>
    bool operator()(const T* x, T* residual) const
    {
        residual[0] = x[0] + 1.0;
        residual[1] = x[1] + 2.0;
        residual[2] = x[2] + 3.0;
        residual[3] = x[3] + 4.0;
        residual[4] = x[4] + 5.0;
        residual[5] = x[5] + 6.0;
        return true;
    }
};

int main()
{
    namespace plt = matplotlibcpp;

    std::mt19937 engine(0);
    std::uniform_real_distribution<> pos_rand(-10, 10);
    std::uniform_real_distribution<> error_rand(-0.1, 0.1);

    constexpr size_t point_cloud_num = 100;

    const Eigen::Vector6d x_truth(5.0, -5.0, 0.5, 0.3, -0.2, -0.3);
    const auto x_truth_transformation = math_utils::lie::exp(x_truth);

    std::array<Eigen::Vector3d, point_cloud_num> point_cloud_a;
    std::array<Eigen::Vector3d, point_cloud_num> point_cloud_b;

    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_a[i] = Eigen::Vector3d(pos_rand(engine), pos_rand(engine), pos_rand(engine));
        point_cloud_b[i] = math_utils::lie::transformation(x_truth_transformation, point_cloud_a[i] + Eigen::Vector3d(error_rand(engine), error_rand(engine), error_rand(engine)));
    }

    auto plot_point_cloud = [](const std::array<Eigen::Vector3d, point_cloud_num>& point_cloud) {
        std::vector<double> x, y, z;
        for (size_t i = 0; i < point_cloud.size(); i++) {
            x.push_back(point_cloud[i](0));
            y.push_back(point_cloud[i](1));
            z.push_back(point_cloud[i](2));
        }
        plt::plot(x, y, ".");
    };

    plot_point_cloud(point_cloud_a);
    plot_point_cloud(point_cloud_b);
    plt::save("a.png");

    Eigen::Vector6d initial_x(5.0, -5.0, 1.0, 1.0, 2.0, 3.0);
    Eigen::Vector6d x = initial_x;

    Problem problem;

    CostFunction* cost_function = new AutoDiffCostFunction<CostFunctor, 6, 6>(new CostFunctor);

    problem.AddResidualBlock(cost_function, NULL, x.data());

    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    std::cout << "x:" << initial_x.transpose() << "->" << x.transpose() << std::endl;
}