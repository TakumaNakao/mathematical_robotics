#pragma once

#include <array>

#include <Eigen/Dense>

#include "math_utils.hpp"

namespace utils
{

class Line {
    // ax + by + c = 0
protected:
    double norm_ab;

public:
    double a;
    double b;
    double c;
    Line() : a(1), b(1), c(0), norm_ab(1){};
    Line(double a_, double b_, double c_) : a(a_), b(b_), c(c_) { norm_ab = std::hypot(a, b); };
    Line(std::array<double, 3> arr) : Line(arr[0], arr[1], arr[2]){};
    Line(Eigen::Vector2d start, Eigen::Vector2d end)
    {
        Eigen::Vector2d diff = (end - start).normalized();
        a = diff(1);
        b = -diff(0);
        c = end(1) * diff(0) - end(0) * diff(1);
        norm_ab = std::hypot(a, b);
    }

    std::optional<double> calc_x(const double y) const
    {
        if (a == 0) {
            return std::nullopt;
        }
        return -(b * y + c) / a;
    }
    std::optional<double> calc_y(const double x) const
    {
        if (b == 0) {
            return std::nullopt;
        }
        return -(a * x + c) / b;
    }
    double angle() const
    {
        double angle = atan2(-a, b);
        if (angle > math_constants::pi / 2) {
            angle -= math_constants::pi;
        }
        if (angle < -math_constants::pi / 2) {
            angle += math_constants::pi;
        }
        return angle;
    }

    virtual double distance(const Eigen::Vector2d& v) const { return fabs(a * v(0) + b * v(1) + c) / norm_ab; }
    virtual double distance(const double x, const double y) const { return distance(Eigen::Vector2d(x, y)); }
    virtual Eigen::Vector2d near(const Eigen::Vector2d& v) const { return v - ((a * v(0) + b * v(1) + c) / norm_ab) * Eigen::Vector2d(a, b); }
    virtual Eigen::Vector2d near(const double x, const double y) const { return near(Eigen::Vector2d(x, y)); }

    virtual void rot(const double theta)
    {
        Eigen::Vector2d ab = math_utils::rotate(Eigen::Vector2d(a, b), theta);
        a = ab(0);
        b = ab(1);
    }
    virtual void rot(const double theta, const Eigen::Vector2d& v)
    {
        move(-v);
        rot(theta);
        move(v);
    }
    virtual void rot(const double theta, const double x, const double y) { rot(theta, Eigen::Vector2d(x, y)); }
    virtual void move(const Eigen::Vector2d& v) { c -= a * v(0) + b * v(1); }
    virtual void move(const double x, const double y) { move(Eigen::Vector2d(x, y)); }
    virtual void rot_move(const Eigen::Vector2d& v, const double theta)
    {
        rot(theta);
        move(v);
    }
    virtual void rot_move(const double x, const double y, const double theta) { rot_move(Eigen::Vector2d(x, y), theta); }

    Line get_rot(const double theta) const
    {
        Line ret = *this;
        ret.rot(theta);
        return ret;
    }
    Line get_rot(const double theta, const Eigen::Vector2d& v) const
    {
        Line ret = *this;
        ret.rot(theta, v);
        return ret;
    }
    Line get_rot(const double theta, const double x, const double y) const { return get_rot(theta, Eigen::Vector2d(x, y)); }
    Line get_move(const Eigen::Vector2d& v) const
    {
        Line ret = *this;
        ret.move(v);
        return ret;
    }
    Line get_move(const double x, const double y) const { return get_move(Eigen::Vector2d(x, y)); }
    Line get_rot_move(const Eigen::Vector2d& v, const double theta) const
    {
        Line ret = get_rot(theta);
        return ret.get_move(v);
    }
    Line get_rot_move(const double x, const double y, const double theta) const { return get_rot_move(Eigen::Vector2d(x, y), theta); }
};
class LineSegment : public Line {
public:
    Eigen::Vector2d start;
    Eigen::Vector2d end;
    LineSegment() : Line(), start(), end(){};
    LineSegment(double a_, double b_, double c_, Eigen::Vector2d start_, Eigen::Vector2d end_) : Line(a_, b_, c_), start(start_), end(end_){};
    LineSegment(std::array<double, 3> arr, Eigen::Vector2d start_, Eigen::Vector2d end_) : Line(arr), start(start_), end(end_){};
    LineSegment(Line line, Eigen::Vector2d start_, Eigen::Vector2d end_) : Line(line), start(start_), end(end_){};
    LineSegment(Eigen::Vector2d start_, Eigen::Vector2d end_) : Line(start_, end_), start(start_), end(end_){};

    double length() const { return (end - start).norm(); }
    bool check_range(const Eigen::Vector2d& v) const
    {
        Eigen::Vector2d p = end - start;
        Eigen::Vector2d q = v - start;
        double r = p.dot(q) / p.dot(p);
        if (r <= 0) {
            return false;
        }
        else if (r >= 1) {
            return false;
        }
        return true;
    }
    bool check_range(const double x, const double y) const { return check_range(Eigen::Vector2d(x, y)); }

    double distance(const Eigen::Vector2d& v) const override
    {
        Eigen::Vector2d p = end - start;
        Eigen::Vector2d q = v - start;
        double r = p.dot(q) / p.dot(p);
        if (r <= 0) {
            return (v - start).norm();
        }
        else if (r >= 1) {
            return (v - end).norm();
        }
        return fabs(a * v(0) + b * v(1) + c) / norm_ab;
    }
    double distance(const double x, const double y) const override { return distance(Eigen::Vector2d(x, y)); }
    Eigen::Vector2d near(const Eigen::Vector2d& v) const override
    {
        Eigen::Vector2d p = end - start;
        Eigen::Vector2d q = v - start;
        double r = p.dot(q) / p.dot(p);
        if (r <= 0) {
            return start;
        }
        else if (r >= 1) {
            return end;
        }
        return v - ((a * v(0) + b * v(1) + c) / norm_ab) * Eigen::Vector2d(a, b);
    }
    Eigen::Vector2d near(const double x, const double y) const override { return near(Eigen::Vector2d(x, y)); }

    void rot(const double theta) override
    {
        Eigen::Vector2d ab = math_utils::rotate(Eigen::Vector2d(a, b), theta);
        a = ab(0);
        b = ab(1);
        start = math_utils::rotate(start, theta);
        end = math_utils::rotate(end, theta);
    }
    void rot(const double theta, const Eigen::Vector2d& v) override
    {
        move(-v);
        rot(theta);
        move(v);
    }
    void rot(const double theta, const double x, const double y) override { rot(theta, Eigen::Vector2d(x, y)); }
    void move(const Eigen::Vector2d& v) override
    {
        c -= a * v(0) + b * v(1);
        start += v;
        end += v;
    }
    void move(const double x, const double y) override { move(Eigen::Vector2d(x, y)); }
    void rot_move(const Eigen::Vector2d& v, const double theta) override
    {
        rot(theta);
        move(v);
    }
    void rot_move(const double x, const double y, const double theta) override { rot_move(Eigen::Vector2d(x, y), theta); }

    LineSegment get_rot(const double theta) const
    {
        LineSegment ret = *this;
        ret.rot(theta);
        return ret;
    }
    LineSegment get_rot(const double theta, const Eigen::Vector2d& v) const
    {
        LineSegment ret = *this;
        ret.rot(theta, v);
        return ret;
    }
    LineSegment get_rot(const double theta, const double x, const double y) const { return get_rot(theta, Eigen::Vector2d(x, y)); }
    LineSegment get_move(const Eigen::Vector2d& v) const
    {
        LineSegment ret = *this;
        ret.move(v);
        return ret;
    }
    LineSegment get_move(const double x, const double y) const { return get_move(Eigen::Vector2d(x, y)); }
    LineSegment get_rot_move(const Eigen::Vector2d& v, const double theta) const
    {
        LineSegment ret = get_rot(theta);
        return ret.get_move(v);
    }
    LineSegment get_rot_move(const double x, const double y, const double theta) const { return get_rot_move(Eigen::Vector2d(x, y), theta); }

    Eigen::Vector2d center(void) const { return (start + end) / 2.0f; }
};

bool operator<(const LineSegment& lhs, const LineSegment& rhs) { return lhs.length() < rhs.length(); }
bool operator>(const LineSegment& lhs, const LineSegment& rhs) { return rhs < lhs; }
bool operator<=(const LineSegment& lhs, const LineSegment& rhs) { return !(lhs > rhs); }
bool operator>=(const LineSegment& lhs, const LineSegment& rhs) { return !(lhs < rhs); }

} // namespace utils