#pragma once

#include <array>
#include <vector>
#include <algorithm>

#include <Eigen/Dense>

namespace Eigen
{
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;
} // namespace Eigen

namespace math_constants
{
constexpr double pi = 3.1415926535897932384626433832795028841971;
constexpr double two_pi = 2 * pi;
} // namespace math_constants

namespace math_utils
{
static inline double normalize_angle_positive(double angle) { return std::fmod(std::fmod(angle, math_constants::two_pi) + math_constants::two_pi, math_constants::two_pi); }

inline double normalize_angle(double angle)
{
    double a = normalize_angle_positive(angle);
    if (a > math_constants::pi)
        a -= math_constants::two_pi;
    return a;
}

static inline double shortest_angular_distance(double from, double to)
{
    double result = normalize_angle_positive(normalize_angle_positive(to) - normalize_angle_positive(from));
    if (result > math_constants::pi) {
        result = -(math_constants::two_pi - result);
    }
    return normalize_angle(result);
}

static inline double deg_to_rad(double deg) { return deg * math_constants::pi / 180.0; }

static inline double rad_to_deg(double rad) { return rad * 180.0 / math_constants::pi; }

template<class T>
static inline T sign(const T& x)
{
    if (x > 0) {
        return T(1);
    }
    else if (x < 0) {
        return T(1);
    }
    else {
        return T(0);
    }
}

template<class T>
static inline T square(const T& x)
{
    return std::pow(x, 2);
}

static inline double sum(const std::vector<double>& v)
{
    double sum = 0;
    for (const auto& x : v) {
        sum += x;
    }
    return sum;
}

static inline double average(const std::vector<double>& v)
{
    assert(v.size() != 0);
    return sum(v) / v.size();
}

static inline double variance(const std::vector<double>& v)
{
    assert(v.size() > 1);
    const double avr = average(v);
    double square_sum = 0;
    for (const auto& x : v) {
        square_sum += square(x - avr);
    }
    return square_sum / (v.size() - 1);
}

static inline double standard_deviation(const std::vector<double>& v) { return std::sqrt(variance(v)); }

static inline Eigen::Vector2d rotate(const Eigen::Vector2d& v, const double theta)
{
    Eigen::Vector2d ret;
    ret(0) = v(0) * std::cos(theta) - v(1) * std::sin(theta);
    ret(1) = v(0) * std::sin(theta) + v(1) * std::cos(theta);
    return ret;
}

} // namespace math_utils