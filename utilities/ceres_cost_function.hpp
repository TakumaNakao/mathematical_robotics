#pragma once

#include <vector>
#include <functional>

#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace ceres_cost_function
{
namespace se3
{

class CostFunction : public ceres::SizedCostFunction<3, 6> {
public:
    static void set_error_func(std::function<Eigen::Vector3d(const Eigen::Vector3d&)> calc_error) { calc_error_ = calc_error; }
    CostFunction(Eigen::Vector3d query) : query_(query) {}
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        Eigen::Vector6d eigen_x = Eigen::Map<const Eigen::Vector6d>(parameters[0]);
        const auto transformation = math_utils::lie::exp(eigen_x);
        auto result = math_utils::lie::transformation(transformation, query_);
        auto e = calc_error_(result);
        residuals[0] = e(0);
        residuals[1] = e(1);
        residuals[2] = e(2);

        if (jacobians != nullptr) {
            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> j(jacobians[0]);
                Eigen::Matrix3d r = transformation.block(0, 0, 3, 3);
                j.block(0, 0, 3, 3) = r;
                j.block(0, 3, 3, 3) = r * math_utils::lie::skew(-query_);
            }
        }
        return true;
    }

private:
    inline static std::function<Eigen::Vector3d(const Eigen::Vector3d&)> calc_error_ = nullptr;
    Eigen::Vector3d query_;
};

class Error {
public:
    template<ceres::NumericDiffMethodType method>
    static ceres::NumericDiffCostFunction<Error, method, 3, 6>* gen_numeric_diff_cost_function(Eigen::Vector3d query)
    {
        return new ceres::NumericDiffCostFunction<Error, method, 3, 6>(new Error(query));
    }
    static void set_error_func(std::function<Eigen::Vector3d(const Eigen::Vector3d&)> calc_error) { calc_error_ = calc_error; }
    Error(Eigen::Vector3d query) : query_(query) {}
    bool operator()(const double* const x, double* residuals) const
    {
        Eigen::Vector6d eigen_x = Eigen::Map<const Eigen::Vector6d>(x);
        const auto transformation = math_utils::lie::exp(eigen_x);
        auto result = math_utils::lie::transformation(transformation, query_);
        auto e = calc_error_(result);
        residuals[0] = e(0);
        residuals[1] = e(1);
        residuals[2] = e(2);
        return true;
    }

private:
    inline static std::function<Eigen::Vector3d(const Eigen::Vector3d&)> calc_error_ = nullptr;
    Eigen::Vector3d query_;
};

} // namespace se3
} // namespace ceres_cost_function