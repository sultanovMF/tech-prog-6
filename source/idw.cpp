#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>

namespace py = pybind11;
using namespace Eigen;
using namespace std;


MatrixXd idw(
    const VectorXd& x_points, const VectorXd& y_points, const VectorXd& values,
    const MatrixXd& x_grid, const MatrixXd& y_grid, double power = 2, double epsilon = 1e-6) {
    
    MatrixXd interpolated_values(x_grid.rows(), x_grid.cols());
    
    for (int i = 0; i < x_grid.rows(); ++i) {
        for (int j = 0; j < x_grid.cols(); ++j) {
            double total_weight = 0;
            double weighted_sum = 0;
            
            for (int k = 0; k < x_points.size(); ++k) {
                double distance_squared = pow(x_points[k] - x_grid(i, j), 2) +
                                         pow(y_points[k] - y_grid(i, j), 2) + epsilon;
                double weight = 1.0 / pow(distance_squared, power / 2);
                weighted_sum += values[k] * weight;
                total_weight += weight;
            }
            
            interpolated_values(i, j) = total_weight > 0 ? weighted_sum / total_weight : 0;
        }
    }
    
    return interpolated_values;
}

PYBIND11_MODULE(idw, m) {
    m.def("idw", &idw, "Inverse Distance Weighting interpolation");
}