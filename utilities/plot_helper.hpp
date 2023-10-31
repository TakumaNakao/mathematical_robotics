#pragma once

#include <vector>

#include <Eigen/Dense>
#include <matplot/matplot.h>

#include "line_segment.hpp"

namespace plot_helper
{

void plot_point_cloud(const std::vector<Eigen::Vector3d>& point_cloud, const std::string& color = "blue", double marker_size = 4)
{
    std::vector<double> x, y, z;
    x.reserve(point_cloud.size());
    y.reserve(point_cloud.size());
    z.reserve(point_cloud.size());
    for (const auto& p : point_cloud) {
        x.push_back(p(0));
        y.push_back(p(1));
        z.push_back(p(2));
    }
    matplot::plot3(x, y, z, ".")->marker_size(marker_size).marker_color(color);
}

void plot_point_cloud_2d(const std::vector<Eigen::Vector3d>& point_cloud, const std::string& color = "blue", double marker_size = 4)
{
    std::vector<double> x, y;
    x.reserve(point_cloud.size());
    y.reserve(point_cloud.size());
    for (const auto& p : point_cloud) {
        x.push_back(p(0));
        y.push_back(p(1));
    }
    matplot::plot(x, y, ".")->marker_size(marker_size).marker_color(color);
}

void plot_line_segment(const std::vector<utils::LineSegment>& line_segment, const std::string& color = "blue", double line_width = 2)
{
    for (const auto& l : line_segment) {
        matplot::line(l.start(0), l.start(1), l.end(0), l.end(1))->line_width(line_width).color(color);
    }
}

} // namespace plot_helper