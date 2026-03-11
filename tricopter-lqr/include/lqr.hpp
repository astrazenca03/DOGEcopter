#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include <Eigen/Dense>

/// Attitude-only LQR controller (inner loop).
///
/// 6-state system:
///   delta_x = [delta_phi, delta_theta, delta_psi, delta_p, delta_q, delta_r]
/// 3 inputs:
///   delta_u = [delta_omega1^2, delta_omega2^2, delta_omega3^2]
///
/// A_att (6x6):  [0_{3x3},  I_{3x3}]    (at hover, omega_0 = 0)
///               [0_{3x3},  0_{3x3}]
///
/// B_att (6x3):  [0_{3x3}        ]
///               [J_inv * B_tau   ]
///
/// Solves CARE iteratively (no external solver).
struct AttitudeLQR {
    Eigen::Matrix<double, 6, 6> A_att;  // linearised A (6x6)
    Eigen::Matrix<double, 6, 3> B_att;  // linearised B (6x3)
    Eigen::Matrix<double, 6, 6> Q;      // state cost (6x6 diagonal)
    Eigen::Matrix3d R;                   // input cost (3x3 diagonal)
    Eigen::Matrix<double, 6, 6> P;      // CARE solution (6x6)
    Eigen::Matrix<double, 3, 6> K;      // gain matrix (3x6)

    /// Build A, B, Q, R from config and allocation, then solve CARE.
    /// Prints gain matrix and closed-loop eigenvalues.
    /// Returns true if all eigenvalues have negative real part.
    bool build(const Config& cfg, const ControlAllocation& alloc);

    /// Compute control: delta_u = -K * delta_x_att
    Eigen::Vector3d compute(const Eigen::Matrix<double, 6, 1>& delta_x) const;

    /// Print diagnostics: K, eigenvalues.
    void printDiagnostics() const;

private:
    /// Solve CARE via iterative Lyapunov method.
    /// Returns true on convergence.
    bool solveCARE();

    /// Solve continuous Lyapunov equation: A^T X + X A = -Q_lyap
    /// Using vectorisation: (I ⊗ A^T + A^T ⊗ I) vec(X) = vec(Q_lyap)
    Eigen::MatrixXd solveLyapunov(const Eigen::MatrixXd& A_cl,
                                   const Eigen::MatrixXd& Q_lyap) const;
};
