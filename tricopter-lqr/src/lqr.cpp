#include "lqr.hpp"
#include <iostream>
#include <Eigen/Eigenvalues>
#include <cmath>

bool AttitudeLQR::build(const Config& cfg, const ControlAllocation& alloc) {
    // --- Build A_att (6x6) ---
    // At hover (omega_0 = 0):
    //   A = [0_{3x3}  I_{3x3}]
    //       [0_{3x3}  0_{3x3}]
    A_att.setZero();
    A_att.block<3,3>(0, 3) = Eigen::Matrix3d::Identity();

    // --- Build B_att (6x3) ---
    // B = [0_{3x3}        ]
    //     [J_inv * B_tau   ]
    Eigen::Matrix3d J_inv = cfg.J.inverse();
    Eigen::Matrix3d B_tau_3x3 = alloc.B_tau;  // (3x3)

    B_att.setZero();
    B_att.block<3,3>(3, 0) = J_inv * B_tau_3x3;

    // --- Build Q (6x6) diagonal ---
    Q.setZero();
    for (int i = 0; i < 6; ++i)
        Q(i, i) = cfg.att_lqr.Q_diag(i);

    // --- Build R (3x3) diagonal ---
    R.setZero();
    for (int i = 0; i < 3; ++i)
        R(i, i) = cfg.att_lqr.R_diag(i);

    std::cout << "\n=== Attitude LQR Setup ===\n";
    std::cout << "A_att (6x6):\n" << A_att << "\n\n";
    std::cout << "B_att (6x3):\n" << B_att << "\n\n";
    std::cout << "J_inv:\n" << J_inv << "\n\n";
    std::cout << "J_inv * B_tau:\n" << J_inv * B_tau_3x3 << "\n\n";

    // --- Solve CARE via Hamiltonian eigenvalue decomposition ---
    bool converged = solveCARE();
    if (!converged) {
        std::cerr << "*** CARE solver failed! ***\n";
        return false;
    }

    // Compute gain: K = R_inv * B^T * P  (3x6)
    K = R.inverse() * B_att.transpose() * P;

    printDiagnostics();

    // Check all closed-loop eigenvalues have negative real part
    Eigen::Matrix<double, 6, 6> A_cl = A_att - B_att * K;
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> es(A_cl);
    bool stable = true;
    for (int i = 0; i < 6; ++i) {
        if (es.eigenvalues()(i).real() >= 0) {
            stable = false;
            break;
        }
    }
    return stable;
}

Eigen::Vector3d AttitudeLQR::compute(const Eigen::Matrix<double, 6, 1>& delta_x) const {
    return -K * delta_x;
}

void AttitudeLQR::printDiagnostics() const {
    std::cout << "\n=== LQR Diagnostics ===\n";
    std::cout << "Gain matrix K_att (3x6):\n" << K << "\n\n";

    Eigen::Matrix<double, 6, 6> A_cl = A_att - B_att * K;
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> es(A_cl);
    std::cout << "Closed-loop eigenvalues of (A - B*K):\n";
    for (int i = 0; i < 6; ++i) {
        auto ev = es.eigenvalues()(i);
        std::cout << "  lambda_" << i << " = " << ev.real();
        if (std::abs(ev.imag()) > 1e-10)
            std::cout << " + " << ev.imag() << "j";
        std::cout << "  (stable: " << (ev.real() < 0 ? "YES" : "NO") << ")\n";
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> pev(P);
    std::cout << "\nP eigenvalues (should all be positive): "
              << pev.eigenvalues().transpose() << "\n";
    std::cout << "======================\n\n";
}

bool AttitudeLQR::solveCARE() {
    // Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
    //
    // Method: Matrix Sign Function applied to the Hamiltonian.
    //
    // The Hamiltonian H has eigenvalues symmetric about the imaginary axis.
    // sign(H) separates the stable/unstable invariant subspaces.
    //
    //   H = [ A,    -S   ]    where S = B R^{-1} B^T
    //       [-Q,    -A^T ]
    //
    // After convergence of sign iteration: sign(H) → W
    //   W = [W11, W12; W21, W22]
    //   P = (W21 - I) * inv(W11 + I)  ... but more robustly:
    //   From the stable subspace: [I; P] spans the columns of 0.5*(I - W)
    //
    // The sign function iteration: Z_{k+1} = 0.5*(Z_k + Z_k^{-1})
    // With determinant scaling for faster convergence.

    const int n = 6;
    const int n2 = 2 * n;  // 12
    Eigen::Matrix3d R_inv = R.inverse();
    Eigen::Matrix<double, 6, 6> S = B_att * R_inv * B_att.transpose();

    // Build Hamiltonian (12x12)
    Eigen::MatrixXd Z(n2, n2);
    Z.block<6,6>(0, 0) =  A_att;
    Z.block<6,6>(0, 6) = -S;
    Z.block<6,6>(6, 0) = -Q;
    Z.block<6,6>(6, 6) = -A_att.transpose();

    // Matrix sign function iteration with determinant scaling
    const int max_iter = 200;
    const double tol = 1e-12;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::MatrixXd Z_inv = Z.inverse();

        // Determinant scaling: gamma = |det(Z)|^{-1/(2n)}
        double det_abs = std::abs(Z.determinant());
        double gamma = 1.0;
        if (det_abs > 1e-300 && std::isfinite(det_abs)) {
            gamma = std::pow(det_abs, -1.0 / n2);
            // Clamp gamma to avoid extreme scaling
            gamma = std::clamp(gamma, 0.1, 10.0);
        }

        Eigen::MatrixXd Z_new = 0.5 * (gamma * Z + Z_inv / gamma);

        double diff = (Z_new - Z).norm();
        Z = Z_new;

        if (diff < tol) {
            std::cout << "[CARE] Sign function converged in " << iter + 1
                      << " iterations (diff: " << diff << ")\n";
            break;
        }
        if (iter == max_iter - 1) {
            std::cout << "[CARE] Sign function: " << max_iter
                      << " iterations, final diff: " << diff << "\n";
        }
    }

    // Extract P from the sign matrix W = sign(H).
    // The stable invariant subspace is spanned by columns of 0.5*(I - W).
    // Partition W = [W11 W12; W21 W22], each 6x6.
    //
    // P = -W21 * (W11 - I)^{-1}  =  (W22 - I)^{-1} * (-W12)
    // More robust: use the relation W21 = -P*(W11+I)/2 ...
    //
    // Standard result: P = (I_n - W22)^{-1} * W21
    // Or equivalently: P = W21 * (I_n - W11)^{-1}   (both should give same result)

    Eigen::MatrixXd W11 = Z.block<6,6>(0, 0);
    Eigen::MatrixXd W21 = Z.block<6,6>(6, 0);
    Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(n, n);

    // Stable projection: Π = (I - W)/2, column space = {[v; Pv]}
    // Π_21 = -W21/2, Π_11 = (I - W11)/2
    // P = Π_21 * Π_11^{-1} = -W21 * (I - W11)^{-1}
    Eigen::MatrixXd P_raw = -W21 * (I_n - W11).inverse();

    // Symmetrise
    P = 0.5 * (P_raw + P_raw.transpose());

    // Verify P is positive definite
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> pev(P);
    double min_eig = pev.eigenvalues().minCoeff();
    if (min_eig < -1e-6) {
        std::cerr << "[CARE] P is not positive definite (min eigenvalue = "
                  << min_eig << ")\n";
        return false;
    }

    // Verify CARE residual: A^T P + P A - P S P + Q ≈ 0
    Eigen::Matrix<double, 6, 6> residual =
        A_att.transpose() * P + P * A_att - P * S * P + Q;
    double res_norm = residual.norm();
    std::cout << "[CARE] Residual norm: " << res_norm << "\n";

    return res_norm < 1e-4;
}

Eigen::MatrixXd AttitudeLQR::solveLyapunov(
    const Eigen::MatrixXd& /* A_cl */,
    const Eigen::MatrixXd& /* Q_lyap */) const
{
    // Kept for interface compatibility — no longer used
    return Eigen::MatrixXd::Zero(6, 6);
}
