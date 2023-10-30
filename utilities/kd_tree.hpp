#pragma once

#include <array>
#include <vector>
#include <limits>
#include <algorithm>

#include <Eigen/Dense>

template<size_t D>
class KdTree {
public:
    static KdTree build(std::vector<Eigen::Matrix<double, D, 1>> p_v) { return KdTree(p_v); }
    KdTree(std::vector<Eigen::Matrix<double, D, 1>>& p_v, size_t depth = 1) : axis_(depth % D)
    {
        const size_t mid = (p_v.size() - 1) / 2;
        std::nth_element(p_v.begin(), p_v.begin() + mid, p_v.end(), [&](const auto& lhs, const auto& rhs) { return lhs(axis_) < rhs(axis_); });
        point_ = p_v[mid];
        {
            std::vector<Eigen::Matrix<double, D, 1>> v(p_v.begin(), p_v.begin() + mid);
            if (!v.empty()) {
                next_node_[0] = std::make_unique<KdTree>(v, depth + 1);
            }
        }
        {
            std::vector<Eigen::Matrix<double, D, 1>> v(p_v.begin() + mid + 1, p_v.end());
            if (!v.empty()) {
                next_node_[1] = std::make_unique<KdTree>(v, depth + 1);
            }
        }
    }
    Eigen::Matrix<double, D, 1> nn_serch(const Eigen::Matrix<double, D, 1>& query, Eigen::Matrix<double, D, 1> guess = {}, std::shared_ptr<double> min_sq_dist = nullptr) const
    {
        if (!min_sq_dist) {
            min_sq_dist = std::make_shared<double>(std::numeric_limits<double>::max());
        }
        const Eigen::Matrix<double, D, 1> e = query - point_;
        const double sq_dist = e.dot(e);
        if (sq_dist < *min_sq_dist) {
            guess = point_;
            *min_sq_dist = sq_dist;
        }
        const size_t dir = query(axis_) < point_(axis_) ? 0 : 1;
        if (next_node_[dir] != nullptr) {
            guess = next_node_[dir]->nn_serch(query, guess, min_sq_dist);
        }
        if (next_node_[!dir] != nullptr && math_utils::square(query(axis_) - point_(axis_)) < *min_sq_dist) {
            guess = next_node_[!dir]->nn_serch(query, guess, min_sq_dist);
        }
        return guess;
    }

private:
    std::array<std::unique_ptr<KdTree>, 2> next_node_ = {nullptr, nullptr};
    Eigen::Matrix<double, D, 1> point_;
    size_t axis_;
};