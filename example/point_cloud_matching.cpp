#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <limits>

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
    TransformationError(Eigen::Vector3d query) : query_(query) {}
    static void set_truth(std::vector<Eigen::Vector3d> truth) { truth_ = truth; }
    bool operator()(const double* const x, double* residual) const
    {
        Eigen::Vector6d eigen_x = Eigen::Map<const Eigen::Vector6d>(x);
        const auto transformation = math_utils::lie::exp(eigen_x);
        auto result = math_utils::lie::transformation(transformation, query_);
        auto e = minimum_error(result);
        residual[0] = e(0);
        residual[1] = e(1);
        residual[2] = e(2);
        return true;
    }

private:
    inline static std::vector<Eigen::Vector3d> truth_ = {};
    Eigen::Vector3d query_;
    Eigen::Vector3d minimum_error(const Eigen::Vector3d& v) const
    {
        double min_e_sq_norm = std::numeric_limits<double>::max();
        Eigen::Vector3d min_e;
        for (const auto& t : truth_) {
            auto e = t - v;
            if (double sq_norm = e.dot(e); sq_norm < min_e_sq_norm) {
                min_e = e;
                min_e_sq_norm = sq_norm;
            }
        }
        return min_e;
    }
};

int main()
{
    namespace plt = matplotlibcpp;

    std::mt19937 engine(0);
    std::uniform_real_distribution<> pos_rand(-10, 10);
    std::uniform_real_distribution<> error_rand(-0.1, 0.1);

    constexpr size_t point_cloud_num = 1000;

    const Eigen::Vector6d x_truth(10.0, -10.0, 0.5, 0.3, -0.2, -0.3);
    const auto x_truth_transformation = math_utils::lie::exp(x_truth);

    std::vector<Eigen::Vector3d> point_cloud_truth(point_cloud_num);
    std::vector<Eigen::Vector3d> point_cloud_query(point_cloud_num);

    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_truth[i] = Eigen::Vector3d(pos_rand(engine), pos_rand(engine), pos_rand(engine));
        point_cloud_query[i] = math_utils::lie::transformation(x_truth_transformation, point_cloud_truth[i] + Eigen::Vector3d(error_rand(engine), error_rand(engine), error_rand(engine)));
    }

    auto plot_point_cloud = [](const std::vector<Eigen::Vector3d>& point_cloud) {
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

    TransformationError::set_truth(point_cloud_truth);

    ceres::Problem problem;
    for (size_t i = 0; i < point_cloud_num; i++) {
        ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<TransformationError, ceres::CENTRAL, 3, 6>(new TransformationError(point_cloud_query[i]));
        problem.AddResidualBlock(cost_function, NULL, x.data());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;
    std::cout << "x:" << initial_x.transpose() << "->" << x.transpose() << std::endl;

    const auto x_transformation = math_utils::lie::exp(x);
    std::vector<Eigen::Vector3d> point_cloud_result(point_cloud_num);
    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_result[i] = math_utils::lie::transformation(x_transformation, point_cloud_query[i]);
    }
    plot_point_cloud(point_cloud_truth);
    plot_point_cloud(point_cloud_result);
    plt::save("point_cloud_matching_after.png");
    plt::clf();
}