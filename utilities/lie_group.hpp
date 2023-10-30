#pragma once

#include "math_utils.hpp"

namespace math_utils::lie
{

constexpr double eps = 1e-5;

static inline Eigen::Matrix3d skew(const Eigen::Vector3d& omega)
{
    Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
    w(0, 1) = -omega(2);
    w(0, 2) = omega(1);
    w(1, 0) = omega(2);
    w(1, 2) = -omega(0);
    w(2, 0) = -omega(1);
    w(2, 1) = omega(0);
    return w;
}

static inline Eigen::Vector3d unskew(const Eigen::Matrix3d& w) { return Eigen::Vector3d(w(2, 1), w(0, 2), w(1, 0)); }

static inline Eigen::Matrix4d make_transformation(const Eigen::Matrix3d& r, const Eigen::Vector3d& t)
{
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Zero();
    transformation.block(0, 0, 3, 3) = r;
    transformation.block(0, 3, 3, 1) = t;
    transformation(3, 3) = 1;
    return transformation;
}

static inline Eigen::Matrix3d exp(const Eigen::Vector3d& omega)
{
    auto w = skew(omega);
    double theta = omega.norm();
    if (theta < eps) {
        return Eigen::Matrix3d::Identity() + w;
    }
    else {
        Eigen::Matrix3d r = w / theta;
        return Eigen::Matrix3d::Identity() + std::sin(theta) * r + (1 - std::cos(theta)) * r * r;
    }
}

static inline Eigen::Vector3d log(const Eigen::Matrix3d& rot)
{
    double tr = rot.trace();
    if (tr + 1 < 1e-3) {
        if (rot(2, 2) > rot(1, 1) && rot(2, 2) > rot(0, 0)) {
            double w = rot(1, 0) - rot(0, 1);
            double q1 = 2 + 2 * rot(2, 2);
            double q2 = rot(2, 0) + rot(0, 2);
            double q3 = rot(1, 2) + rot(2, 1);
            double norm = std::sqrt(square(q1) + square(q2) + square(q3) + square(w));
            double sign_w = sign(w);
            double magnitude = math_constants::pi - (2 * sign_w * w) / norm;
            double scale = 0.5 * magnitude / std::sqrt(q1);
            return sign_w * scale * Eigen::Vector3d(q2, q3, q1);
        }
        else if (rot(1, 1) > rot(0, 0)) {
            double w = rot(0, 2) - rot(2, 0);
            double q1 = 2 + 2 * rot(1, 1);
            double q2 = rot(1, 2) + rot(2, 1);
            double q3 = rot(0, 1) + rot(1, 0);
            double norm = std::sqrt(square(q1) + square(q2) + square(q3) + square(w));
            double sign_w = sign(w);
            double magnitude = math_constants::pi - (2 * sign_w * w) / norm;
            double scale = 0.5 * magnitude / std::sqrt(q1);
            return sign_w * scale * Eigen::Vector3d(q3, q1, q2);
        }
        else {
            double w = rot(2, 1) - rot(1, 2);
            double q1 = 2 + 2 * rot(0, 0);
            double q2 = rot(0, 1) + rot(1, 0);
            double q3 = rot(2, 0) + rot(0, 2);
            double norm = std::sqrt(square(q1) + square(q2) + square(q3) + square(w));
            double sign_w = sign(w);
            double magnitude = math_constants::pi - (2 * sign_w * w) / norm;
            double scale = 0.5 * magnitude / std::sqrt(q1);
            return sign_w * scale * Eigen::Vector3d(q1, q2, q3);
        }
    }
    else {
        double magnitude;
        double tr3 = tr - 3;
        if (tr3 < -1e-6) {
            double theta = std::acos(0.5 * (tr - 1));
            magnitude = theta / (2 * std::sin(theta));
        }
        else {
            magnitude = 0.5 - tr3 / 12 + square(tr3) / 60;
        }
        return magnitude * Eigen::Vector3d(rot(2, 1) - rot(1, 2), rot(0, 2) - rot(2, 0), rot(1, 0) - rot(0, 1));
    }
}

static inline Eigen::Matrix4d exp(const Eigen::Vector6d& x)
{
    Eigen::Vector3d v = x.head(3);
    Eigen::Vector3d omega = x.tail(3);
    auto rot = exp(omega);
    double theta2 = omega.dot(omega);
    if (theta2 < eps) {
        return make_transformation(rot, v);
    }
    else {
        Eigen::Vector3d t_parallel = omega * omega.dot(v);
        Eigen::Vector3d omega_cross_v = omega.cross(v);
        Eigen::Vector3d t = (omega_cross_v - rot * omega_cross_v + t_parallel) / theta2;
        return make_transformation(rot, t);
    }
}

static inline Eigen::Vector6d log(const Eigen::Matrix4d& transformation)
{
    Eigen::Vector3d t = transformation.block(0, 3, 3, 1);
    Eigen::Matrix3d rot = transformation.block(0, 0, 3, 3);
    auto omega = log(rot);
    double theta = omega.norm();
    if (theta < 1e-10) {
        return Eigen::Vector6d(t(0), t(1), t(2), omega(0), omega(1), omega(2));
    }
    else {
        auto w = skew(omega / theta);
        Eigen::Vector3d wt = w * t;
        Eigen::Vector3d u = t - (0.5 * theta) * wt + (1 - theta / (2 * std::tan(0.5 * theta))) * w * wt;
        return Eigen::Vector6d(u(0), u(1), u(2), omega(0), omega(1), omega(2));
    }
}

static inline Eigen::Vector3d transformation(const Eigen::Matrix4d& transformation, const Eigen::Vector3d& p)
{
    Eigen::Vector4d ph;
    ph << p, 1.0;
    return (transformation * ph).head(3);
}

} // namespace math_utils::lie