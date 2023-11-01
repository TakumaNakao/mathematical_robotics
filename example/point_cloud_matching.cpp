#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <limits>
#include <functional>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <matplot/matplot.h>

#include "math_utils.hpp"
#include "lie_group.hpp"
#include "kd_tree.hpp"
#include "ceres_cost_function.hpp"
#include "gauss_newton_method.hpp"
#include "plot_helper.hpp"

namespace plt = matplot;

namespace solve_mode_constant
{
const int CERES_ANALYTIC_DERIVATIVES = 1;
const int CERES_NUMERIC_DERIVATIVES = 2;
const int SELF_GAUSS_NEWTON = 3;
} // namespace solve_mode_constant

int main()
{
    std::cout << "select solve mode" << std::endl;
    std::cout << "  " << solve_mode_constant::CERES_ANALYTIC_DERIVATIVES << ": Ceres Analytic Derivatives" << std::endl;
    std::cout << "  " << solve_mode_constant::CERES_NUMERIC_DERIVATIVES << ": Ceres Numeric Derivatives" << std::endl;
    std::cout << "  " << solve_mode_constant::SELF_GAUSS_NEWTON << ": Self Gauss Newton" << std::endl;
    int solve_mode;
    std::cin >> solve_mode;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    if (solve_mode != solve_mode_constant::CERES_ANALYTIC_DERIVATIVES && solve_mode != solve_mode_constant::CERES_NUMERIC_DERIVATIVES && solve_mode != solve_mode_constant::SELF_GAUSS_NEWTON) {
        std::exit(-1);
    }

    std::mt19937 engine(0);
    std::uniform_real_distribution<> pos_rand(-10, 10);
    std::uniform_real_distribution<> error_rand(-0.1, 0.1);

    constexpr size_t point_cloud_num = 300;

    const Eigen::Vector6d x_truth(15.0, -15.0, 10.0, 0.3, -0.2, -0.3);
    const auto x_truth_transformation = math_utils::lie::exp(x_truth);

    std::vector<Eigen::Vector3d> point_cloud_truth(point_cloud_num);
    std::vector<Eigen::Vector3d> point_cloud_query(point_cloud_num);

    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_truth[i] = Eigen::Vector3d(pos_rand(engine), pos_rand(engine), pos_rand(engine));
        point_cloud_query[i] = math_utils::lie::transformation(x_truth_transformation, point_cloud_truth[i] + Eigen::Vector3d(error_rand(engine), error_rand(engine), error_rand(engine)));
    }

    plt::hold(plt::on);
    plot_helper::plot_point_cloud(point_cloud_truth, "blue");
    plot_helper::plot_point_cloud(point_cloud_query, "red");
    plt::hold(plt::off);
    plt::save("img/point_cloud_matching_before.png");
    plt::cla();

    auto point_cloud_truth_tree = std::make_shared<KdTree<3>>(KdTree<3>::build(point_cloud_truth));
    auto calc_error = [point_cloud_truth_tree](const Eigen::Vector3d& v) -> Eigen::Vector3d { return v - point_cloud_truth_tree->nn_serch(v); };

    Eigen::Vector6d initial_x = Eigen::Vector6d::Zero();
    Eigen::Vector6d x = initial_x;

    if (solve_mode == solve_mode_constant::CERES_ANALYTIC_DERIVATIVES || solve_mode == solve_mode_constant::CERES_NUMERIC_DERIVATIVES) {
        ceres_cost_function::se3::Error::set_error_func(calc_error);
        ceres_cost_function::se3::CostFunction::set_error_func(calc_error);

        ceres::Problem problem;
        for (size_t i = 0; i < point_cloud_num; i++) {
            ceres::CostFunction* cost_function;
            if (solve_mode == solve_mode_constant::CERES_ANALYTIC_DERIVATIVES) {
                cost_function = new ceres_cost_function::se3::CostFunction(point_cloud_query[i]);
            }
            else {
                cost_function = new ceres::NumericDiffCostFunction<ceres_cost_function::se3::Error, ceres::CENTRAL, 3, 6>(new ceres_cost_function::se3::Error(point_cloud_query[i]));
            }
            problem.AddResidualBlock(cost_function, nullptr, x.data());
        }

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << std::endl;
    }
    else if (solve_mode == solve_mode_constant::SELF_GAUSS_NEWTON) {
        auto plot = [&](const Eigen::Vector6d& x, size_t i) {
            std::vector<Eigen::Vector3d> transformed_query;
            const auto transformation = math_utils::lie::exp(x);
            for (const auto& q : point_cloud_query) {
                transformed_query.push_back(math_utils::lie::transformation(transformation, q));
            }
            plt::hold(plt::on);
            plot_helper::plot_point_cloud(point_cloud_truth, "blue");
            plot_helper::plot_point_cloud(transformed_query, "red");
            plt::hold(plt::off);
            plt::save("img/" + std::to_string(i) + ".png");
            plt::cla();
        };
        x = gauss_newton_method::se3::solve(initial_x, point_cloud_query, calc_error, plot);
    }

    std::cout << "x:" << initial_x.transpose() << "->" << x.transpose() << std::endl;

    const auto x_transformation = math_utils::lie::exp(x);
    std::vector<Eigen::Vector3d> point_cloud_result(point_cloud_num);
    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_result[i] = math_utils::lie::transformation(x_transformation, point_cloud_query[i]);
    }
    plt::hold(plt::on);
    plot_helper::plot_point_cloud(point_cloud_truth, "blue");
    plot_helper::plot_point_cloud(point_cloud_result, "red");
    plt::hold(plt::off);
    plt::save("img/point_cloud_matching_after.png");
    plt::show();
    plt::cla();
}
