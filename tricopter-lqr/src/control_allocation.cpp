#include "control_allocation.hpp"
#include <iostream>
#include <Eigen/Eigenvalues>

void ControlAllocation::build(const Config& cfg) {
    const int N = static_cast<int>(cfg.rotors.size());
    B_alloc.resize(6, N);   // (6 x N)
    B_force.resize(3, N);   // (3 x N)
    B_tau.resize(3, N);     // (3 x N)

    for (int i = 0; i < N; ++i) {
        const auto& r = cfg.rotors[i];

        // Force column: k_T * thrust_axis
        Eigen::Vector3d f_col = r.k_T * r.thrust_axis;

        // Torque column: cross(position, k_T * thrust_axis) + d * k_Q * spin_axis
        Eigen::Vector3d tau_col = r.position.cross(f_col)
            + static_cast<double>(r.spin_direction) * r.k_Q * r.spin_axis;

        B_alloc.col(i).head<3>() = f_col;    // rows 0-2: force
        B_alloc.col(i).tail<3>() = tau_col;   // rows 3-5: torque

        B_force.col(i) = f_col;
        B_tau.col(i) = tau_col;
    }
}

void ControlAllocation::printDiagnostics() const {
    std::cout << "\n=== Control Allocation Diagnostics ===\n";
    std::cout << "B_alloc (6 x " << B_alloc.cols() << "):\n" << B_alloc << "\n\n";
    std::cout << "B_tau (3 x " << B_tau.cols() << "):\n" << B_tau << "\n\n";

    // For square B_tau (3x3), compute rank, condition number, eigenvalues
    if (B_tau.cols() == 3) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_tau);
        auto sv = svd.singularValues();
        int rank = 0;
        for (int i = 0; i < sv.size(); ++i)
            if (sv(i) > 1e-10) ++rank;

        double cond = (sv(sv.size()-1) > 1e-15) ? sv(0) / sv(sv.size()-1) : 1e15;

        std::cout << "Singular values of B_tau: " << sv.transpose() << "\n";
        std::cout << "Rank of B_tau: " << rank << "\n";
        std::cout << "Condition number of B_tau: " << cond << "\n";

        if (rank < 3)
            std::cout << "*** WARNING: B_tau rank < 3 — uncontrollable torque axis! ***\n";

        // Eigenvalues of B_tau (for square matrix)
        Eigen::EigenSolver<Eigen::MatrixXd> es(B_tau);
        std::cout << "Eigenvalues of B_tau:\n" << es.eigenvalues() << "\n";
    } else {
        // Non-square: use SVD rank
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_tau);
        auto sv = svd.singularValues();
        int rank = 0;
        for (int i = 0; i < sv.size(); ++i)
            if (sv(i) > 1e-10) ++rank;
        std::cout << "Rank of B_tau: " << rank << " (non-square " << B_tau.rows()
                  << "x" << B_tau.cols() << ")\n";
    }
    std::cout << "======================================\n\n";
}
