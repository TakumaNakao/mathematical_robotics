#pragma once

#include <vector>
#include <chrono>
#include <limits>
#include <functional>

#include <Eigen/Dense>

namespace gauss_newton_method
{
namespace se3
{

Eigen::Vector6d solve(const Eigen::Vector6d& init_x, const std::vector<Eigen::Vector3d>& query, const std::function<Eigen::Vector3d(const Eigen::Vector3d&)>& calc_error, std::function<void(const Eigen::Vector6d&, size_t)> callback = nullptr)
{
    auto start = std::chrono::system_clock::now();
    auto solve_once = [&query, &calc_error](const Eigen::Vector6d& x) -> std::tuple<Eigen::Vector6d, double> {
        Eigen::Matrix6d h = Eigen::Matrix6d::Zero();
        Eigen::Vector6d g = Eigen::Vector6d::Zero();
        double cost = 0;
        const auto transformation = math_utils::lie::exp(x);
        for (size_t j = 0; j < query.size(); j++) {
            auto result = math_utils::lie::transformation(transformation, query[j]);
            auto e = calc_error(result);
            Eigen::Matrix3d r = transformation.block(0, 0, 3, 3);
            Eigen::Matrix<double, 3, 6> jac;
            jac.block(0, 0, 3, 3) = transformation.block(0, 0, 3, 3);
            jac.block(0, 3, 3, 3) = transformation.block(0, 0, 3, 3) * math_utils::lie::skew(-query[j]);
            h += jac.transpose() * jac;
            g += jac.transpose() * e;
            cost += 0.5 * e.dot(e);
        }
        Eigen::Vector6d dx = h.fullPivLu().solve(-g);
        return {dx, cost};
    };
    double last_cost = std::numeric_limits<double>::max();
    Eigen::Vector6d x = init_x;
    size_t count = 0;
    while (true) {
        auto [dx, cost] = solve_once(x);
        x += dx;
        if (callback) {
            callback(x, count);
        }
        std::cout << "itr: " << count << " , cost: " << cost << std::endl;
        if (last_cost < cost) {
            break;
        }
        if (last_cost - cost < 1e-6) {
            break;
        }
        last_cost = cost;
        count++;
    }
    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
    std::cout << "time: " << elapsed << "[s]" << std::endl;
    return x;
}

} // namespace se3
} // namespace gauss_newton_method