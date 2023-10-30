#pragma once

#include <vector>

#include <Eigen/Dense>
#include <matplot/matplot.h>

namespace plot_helper
{

void plot_point_cloud(const std::vector<Eigen::Vector3d>& point_cloud, const std::string& color = "blue", double marker_size = 4)
{
    std::vector<double> x, y, z;
    x.reserve(point_cloud.size());
    y.reserve(point_cloud.size());
    z.reserve(point_cloud.size());
    for (size_t i = 0; i < point_cloud.size(); i++) {
        x.push_back(point_cloud[i](0));
        y.push_back(point_cloud[i](1));
        z.push_back(point_cloud[i](2));
    }
    matplot::plot3(x, y, z, ".")->marker_size(marker_size).color(color);
}

} // namespace plot_helper