#include <iostream>
#include <fstream>
#include <vector>

#include "fast_matrix_market/fast_matrix_market.hpp"
#include "fast_matrix_market/app/Eigen.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>

#define GREEN "\033[1;32m"
#define RED "\033[1;31m"
#define RESET "\033[0m"

namespace fmm = fast_matrix_market;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " matrix.mtx in_vector.mtx out_vector.mtx\n";
        return 1;
    }

    std::ifstream mat_file(argv[1]);
    std::ifstream in_vec_file(argv[2]);
    std::ifstream out_vec_file(argv[3]);

    Eigen::SparseMatrix<double> mat;
    Eigen::VectorXd x;
    Eigen::VectorXd y;
    // Load matrix and vector data
    fmm::read_matrix_market_eigen(mat_file, mat);
    fmm::read_matrix_market_eigen_dense(in_vec_file, x);
    fmm::read_matrix_market_eigen_dense(out_vec_file, y);

    // Calculate product
    Eigen::VectorXd res = mat * x;

    // Check result and error
    const double epsilon = 1e-4;
    std::cout << "Norm: " << (res - y).norm() << "\n";
    if ((res - y).norm() < epsilon) {
        std::cout << GREEN << "Results matches, the matrix-vector product is correct!!!\n"
                  << RESET;
    } else {
        std::cout << RED << "Results differs, the matrix-vector product is not correct!!!\n"
                  << RESET;

        std::ofstream res_vec_file("result.mtx");
        std::ofstream err_vec_file("error.mtx");
        fmm::write_matrix_market_eigen_dense(res_vec_file, res);
        fmm::write_matrix_market_eigen_dense(err_vec_file, y - res);
    }

    return 0;
}
