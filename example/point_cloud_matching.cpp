#include <iostream>
#include <array>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <limits>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <matplot/matplot.h>

#include "math_utils.hpp"
#include "lie_group.hpp"
#include "kd_tree.hpp"

namespace plt = matplot;

void plot_point_cloud(const std::vector<Eigen::Vector3d>& point_cloud, const std::string& color = "blue")
{
    std::vector<double> x, y, z;
    for (size_t i = 0; i < point_cloud.size(); i++) {
        x.push_back(point_cloud[i](0));
        y.push_back(point_cloud[i](1));
        z.push_back(point_cloud[i](2));
    }
    plt::plot3(x, y, z, ".")->marker_size(4).color(color);
}

Eigen::Vector6d solve(const Eigen::Vector6d& init_x, const std::vector<Eigen::Vector3d>& query, const std::shared_ptr<KdTree<3>> truth, const std::vector<Eigen::Vector3d>& truth_vector = {})
{
    auto plot = [&](const Eigen::Vector6d& x, size_t i) {
        if (truth_vector.empty()) {
            return;
        }
        std::vector<Eigen::Vector3d> transformed_query;
        const auto transformation = math_utils::lie::exp(x);
        for (const auto& q : query) {
            transformed_query.push_back(math_utils::lie::transformation(transformation, q));
        }
        plt::hold(plt::on);
        plot_point_cloud(truth_vector, "blue");
        plot_point_cloud(transformed_query, "red");
        plt::hold(plt::off);
        plt::save("img/" + std::to_string(i) + ".png");
        plt::cla();
    };
    size_t plot_count = 0;
    auto start = std::chrono::system_clock::now();
    auto solve_once = [&query, &truth](const Eigen::Vector6d& x) -> std::tuple<Eigen::Vector6d, double> {
        Eigen::Matrix6d h = Eigen::Matrix6d::Zero();
        Eigen::Vector6d g = Eigen::Vector6d::Zero();
        double cost = 0;
        const auto transformation = math_utils::lie::exp(x);
        for (size_t j = 0; j < query.size(); j++) {
            auto result = math_utils::lie::transformation(transformation, query[j]);
            auto nn = truth->nn_serch(result);
            auto e = result - nn;
            Eigen::Matrix3d r = transformation.block(0, 0, 3, 3);
            Eigen::Matrix<double, 3, 6> jac;
            jac.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
            jac.block(0, 3, 3, 3) = math_utils::lie::skew(-result);
            h += jac.transpose() * jac;
            g += jac.transpose() * e;
            cost += 0.5 * e.dot(e);
        }
        Eigen::Vector6d dx = h.fullPivLu().solve(-g);
        return {dx, cost};
    };
    double last_cost = std::numeric_limits<double>::max();
    Eigen::Vector6d x = init_x;
    while (true) {
        auto [dx, cost] = solve_once(x);
        x = math_utils::lie::log(Eigen::Matrix4d(math_utils::lie::exp(x) * math_utils::lie::exp(dx)));
        plot(x, plot_count++);
        std::cout << "cost: " << cost << std::endl;
        if (last_cost < cost) {
            break;
        }
        if (last_cost - cost < 1e-6) {
            break;
        }
        last_cost = cost;
    }
    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "time: " << elapsed << std::endl;
    return x;
}

class TransformationCostFunction : public ceres::SizedCostFunction<3, 6> {
public:
    static void set_truth(std::shared_ptr<KdTree<3>> truth) { truth_ = truth; }
    TransformationCostFunction(Eigen::Vector3d query) : query_(query) {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Eigen::Vector6d eigen_x = Eigen::Map<const Eigen::Vector6d>(parameters[0]);
        const auto transformation = math_utils::lie::exp(eigen_x);
        auto result = math_utils::lie::transformation(transformation, query_);
        auto nn = truth_->nn_serch(result);
        auto e = result - nn;
        residuals[0] = e(0);
        residuals[1] = e(1);
        residuals[2] = e(2);

        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> j(jacobians[0]);
                Eigen::Matrix3d r = transformation.block(0, 0, 3, 3);
                j.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
                j.block(0, 3, 3, 3) = math_utils::lie::skew(-result);
            }
        }
        return true;
    }

private:
    inline static std::shared_ptr<KdTree<3>> truth_ = nullptr;
    Eigen::Vector3d query_;
};

class TransformationError {
public:
    static void set_truth(std::shared_ptr<KdTree<3>> truth) { truth_ = truth; }
    TransformationError(Eigen::Vector3d query) : query_(query) {}
    bool operator()(const double* const x, double* residuals) const
    {
        Eigen::Vector6d eigen_x = Eigen::Map<const Eigen::Vector6d>(x);
        const auto transformation = math_utils::lie::exp(eigen_x);
        auto result = math_utils::lie::transformation(transformation, query_);
        auto nn = truth_->nn_serch(result);
        auto e = result - nn;
        residuals[0] = e(0);
        residuals[1] = e(1);
        residuals[2] = e(2);
        return true;
    }

private:
    inline static std::shared_ptr<KdTree<3>> truth_ = nullptr;
    Eigen::Vector3d query_;
};

int main()
{
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
    plot_point_cloud(point_cloud_truth, "blue");
    plot_point_cloud(point_cloud_query, "red");
    plt::hold(plt::off);
    plt::save("img/point_cloud_matching_before.png");
    plt::show();
    plt::cla();

    auto point_cloud_truth_tree = std::make_shared<KdTree<3>>(KdTree<3>::build(point_cloud_truth));

    Eigen::Vector6d initial_x = Eigen::Vector6d::Zero();
    Eigen::Vector6d x = initial_x;

    TransformationError::set_truth(point_cloud_truth_tree);
    TransformationCostFunction::set_truth(point_cloud_truth_tree);

    ceres::Problem problem;
    for (size_t i = 0; i < point_cloud_num; i++) {
        // ceres::CostFunction* cost_function = new ceres::NumericDiffCostFunction<TransformationError, ceres::CENTRAL, 3, 6>(new TransformationError(point_cloud_query[i]));
        ceres::CostFunction* cost_function = new TransformationCostFunction(point_cloud_query[i]);
        problem.AddResidualBlock(cost_function, NULL, x.data());
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    // x = solve(initial_x, point_cloud_query, point_cloud_truth_tree, point_cloud_truth);

    std::cout << "x:" << initial_x.transpose() << "->" << x.transpose() << std::endl;

    const auto x_transformation = math_utils::lie::exp(x);
    std::vector<Eigen::Vector3d> point_cloud_result(point_cloud_num);
    for (size_t i = 0; i < point_cloud_num; i++) {
        point_cloud_result[i] = math_utils::lie::transformation(x_transformation, point_cloud_query[i]);
    }
    plt::hold(plt::on);
    plot_point_cloud(point_cloud_truth, "blue");
    plot_point_cloud(point_cloud_result, "red");
    plt::hold(plt::off);
    plt::save("img/point_cloud_matching_after.png");
    plt::show();
    plt::cla();
}
