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
#include "line_segment.hpp"
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

    const Eigen::Vector6d x_truth(2.0, -4.0, 0.0, 0.0, 0.0, 0.3);
    const auto x_truth_transformation = math_utils::lie::exp(x_truth);

    std::vector<utils::LineSegment> map_line_segment;
    map_line_segment.push_back(utils::LineSegment(Eigen::Vector2d(10, -2), Eigen::Vector2d(10, 6)));
    map_line_segment.push_back(utils::LineSegment(Eigen::Vector2d(10, 8), Eigen::Vector2d(10, 12)));
    map_line_segment.push_back(utils::LineSegment(Eigen::Vector2d(12, -2), Eigen::Vector2d(12, 12)));
    map_line_segment.push_back(utils::LineSegment(Eigen::Vector2d(0, 6), Eigen::Vector2d(10, 6)));
    map_line_segment.push_back(utils::LineSegment(Eigen::Vector2d(0, 8), Eigen::Vector2d(10, 8)));

    std::mt19937 engine(0);
    std::uniform_real_distribution<> error_rand(-0.1, 0.1);
    std::vector<Eigen::Vector3d> point_cloud_query;
    {
        std::uniform_real_distribution<> rand(3, 6);
        for (size_t i = 0; i < 25; i++) {
            Eigen::Vector3d pos(10, rand(engine), 0);
            Eigen::Vector3d error(error_rand(engine), error_rand(engine), 0);
            point_cloud_query.push_back(math_utils::lie::transformation(x_truth_transformation, pos + error));
        }
    }
    {
        std::uniform_real_distribution<> rand(8, 11);
        for (size_t i = 0; i < 25; i++) {
            Eigen::Vector3d pos(10, rand(engine), 0);
            Eigen::Vector3d error(error_rand(engine), error_rand(engine), 0);
            point_cloud_query.push_back(math_utils::lie::transformation(x_truth_transformation, pos + error));
        }
    }
    {
        std::uniform_real_distribution<> rand(2, 10);
        for (size_t i = 0; i < 70; i++) {
            Eigen::Vector3d pos(rand(engine), 6, 0);
            Eigen::Vector3d error(error_rand(engine), error_rand(engine), 0);
            point_cloud_query.push_back(math_utils::lie::transformation(x_truth_transformation, pos + error));
        }
    }
    {
        std::uniform_real_distribution<> rand(2, 10);
        for (size_t i = 0; i < 70; i++) {
            Eigen::Vector3d pos(rand(engine), 8, 0);
            Eigen::Vector3d error(error_rand(engine), error_rand(engine), 0);
            point_cloud_query.push_back(math_utils::lie::transformation(x_truth_transformation, pos + error));
        }
    }

    plt::hold(plt::on);
    plot_helper::plot_line_segment(map_line_segment, "blue");
    plot_helper::plot_point_cloud(point_cloud_query, "red");
    plt::hold(plt::off);
    plt::save("img/point_cloud_to_line_matching_before.png");
    plt::cla();

    auto calc_error = [&map_line_segment](const Eigen::Vector3d& v) -> Eigen::Vector3d {
        size_t min_idx = 0;
        double min_distance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < map_line_segment.size(); i++) {
            if (auto d = map_line_segment[i].distance(v.head(2)); d < min_distance) {
                min_idx = i;
                min_distance = d;
            }
        }
        auto e = v.head(2) - map_line_segment[min_idx].near(v.head(2));
        return Eigen::Vector3d(e(0), e(1), 0);
    };

    Eigen::Vector6d initial_x = Eigen::Vector6d::Zero();
    Eigen::Vector6d x = initial_x;

    if (solve_mode == solve_mode_constant::CERES_ANALYTIC_DERIVATIVES || solve_mode == solve_mode_constant::CERES_NUMERIC_DERIVATIVES) {
        ceres_cost_function::se3::Error::set_error_func(calc_error);
        ceres_cost_function::se3::CostFunction::set_error_func(calc_error);

        ceres::Problem problem;
        for (size_t i = 0; i < point_cloud_query.size(); i++) {
            ceres::CostFunction* cost_function;
            if (solve_mode == solve_mode_constant::CERES_ANALYTIC_DERIVATIVES) {
                cost_function = new ceres_cost_function::se3::CostFunction(point_cloud_query[i]);
            }
            else {
                cost_function = ceres_cost_function::se3::Error::gen_numeric_diff_cost_function<ceres::CENTRAL>(point_cloud_query[i]);
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
            plot_helper::plot_line_segment(map_line_segment, "blue");
            plot_helper::plot_point_cloud_2d(transformed_query, "red");
            plt::hold(plt::off);
            plt::save("img/" + std::to_string(i) + ".png");
            plt::cla();
        };
        x = gauss_newton_method::se3::solve(initial_x, point_cloud_query, calc_error, plot);
    }

    std::cout << "x:" << initial_x.transpose() << "->" << x.transpose() << std::endl;

    const auto x_transformation = math_utils::lie::exp(x);
    std::vector<Eigen::Vector3d> point_cloud_result(point_cloud_query.size());
    for (size_t i = 0; i < point_cloud_query.size(); i++) {
        point_cloud_result[i] = math_utils::lie::transformation(x_transformation, point_cloud_query[i]);
    }
    plt::hold(plt::on);
    plot_helper::plot_line_segment(map_line_segment, "blue");
    plot_helper::plot_point_cloud_2d(point_cloud_result, "red");
    plt::hold(plt::off);
    plt::save("img/point_cloud_to_line_matching_after.png");
    plt::show();
    plt::cla();
}
