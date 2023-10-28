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

// class TransformationError : public ceres::SizedCostFunction<3, 6> {
// public:
//     ReprojectionErrorSE3XYZ(Eigen::Vector3d ref) : ref_(ref) {}

//     virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

// private:
//     Eigen::Vector3d ref_;
// };

class TransformationError {
public:
    TransformationError(Eigen::Vector3d query, Eigen::Vector3d truth) : query_(query), truth_(truth) {}
    bool operator()(const double* const x, double* residual) const
    {
        Eigen::Vector6d eigen_x = Eigen::Map<const Eigen::Vector6d>(x);
        const auto transformation = math_utils::lie::exp(eigen_x);
        auto result = math_utils::lie::transformation(transformation, query_);
        auto e = truth_ - result;
        residual[0] = e(0);
        residual[1] = e(1);
        residual[2] = e(2);
        return true;
    }

private:
    Eigen::Vector3d query_;
    Eigen::Vector3d truth_;
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

    std::array<Eigen::Vector3d, point_cloud_num> point_cloud_truth;
    std::array<Eigen::Vector3d, point_cloud_num> point_cloud_query;

    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_truth[i] = Eigen::Vector3d(pos_rand(engine), pos_rand(engine), pos_rand(engine));
        point_cloud_query[i] = math_utils::lie::transformation(x_truth_transformation, point_cloud_truth[i] + Eigen::Vector3d(error_rand(engine), error_rand(engine), error_rand(engine)));
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

    plot_point_cloud(point_cloud_truth);
    plot_point_cloud(point_cloud_query);
    plt::save("point_cloud_matching_before.png");
    plt::clf();

    Eigen::Vector6d initial_x = Eigen::Vector6d::Zero();
    Eigen::Vector6d x = initial_x;

    ceres::Problem problem;
    for (size_t i = 0; i < point_cloud_num; i++) {
        ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<TransformationError, ceres::CENTRAL, 3, 6>(new TransformationError(point_cloud_query[i], point_cloud_truth[i]));
        problem.AddResidualBlock(cost_function, NULL, x.data());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    std::cout << "x:" << initial_x.transpose() << "->" << x.transpose() << std::endl;

    const auto x_transformation = math_utils::lie::exp(x);
    std::array<Eigen::Vector3d, point_cloud_num> point_cloud_result;
    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_result[i] = math_utils::lie::transformation(x_transformation, point_cloud_query[i]);
    }
    plot_point_cloud(point_cloud_truth);
    plot_point_cloud(point_cloud_result);
    plt::save("point_cloud_matching_after.png");
    plt::clf();
}