#include "lqr.hpp"
#include <iostream>
#include <Eigen/Eigenvalues>
#include <cmath>

// ============================================================================
// RollPitchLQR
// ============================================================================

bool RollPitchLQR::build(const Config& cfg, const ControlAllocation& alloc) {
    // --- Build A_rp (4x4) ---
    // At hover (omega_0 = 0):
    //   A = [0_{2x2}  I_{2x2}]
    //       [0_{2x2}  0_{2x2}]
    A_rp.setZero();
    A_rp.block<2,2>(0, 2) = Eigen::Matrix2d::Identity();

    // --- Build B_rp (4x3) ---
    // B = [0_{2x3}                          ]
    //     [first 2 rows of J_inv * B_tau    ]
    Eigen::Matrix3d J_inv = cfg.J.inverse();
    Eigen::Matrix3d B_tau_3x3 = alloc.B_tau;  // (3x3)
    Eigen::Matrix3d JinvBtau = J_inv * B_tau_3x3;

    B_rp.setZero();
    B_rp.block<2,3>(2, 0) = JinvBtau.topRows<2>();  // rows 0,1 of J_inv*B_tau

    // --- Build Q (4x4) diagonal ---
    Q.setZero();
    for (int i = 0; i < 4; ++i)
        Q(i, i) = cfg.att_lqr.Q_diag(i);

    // --- Build R (3x3) diagonal ---
    R.setZero();
    for (int i = 0; i < 3; ++i)
        R(i, i) = cfg.att_lqr.R_diag(i);

    std::cout << "\n=== Roll/Pitch LQR Setup (4-state) ===\n";
    std::cout << "A_rp (4x4):\n" << A_rp << "\n\n";
    std::cout << "B_rp (4x3):\n" << B_rp << "\n\n";
    std::cout << "J_inv:\n" << J_inv << "\n\n";
    std::cout << "J_inv * B_tau (full 3x3):\n" << JinvBtau << "\n";
    std::cout << "  (using rows 0-1 for roll/pitch)\n\n";

    // --- Solve CARE ---
    bool converged = solveCARE();
    if (!converged) {
        std::cerr << "*** CARE solver failed! ***\n";
        return false;
    }

    // Compute gain: K = R_inv * B^T * P  (3x4)
    K = R.inverse() * B_rp.transpose() * P;

    printDiagnostics();

    // Check all closed-loop eigenvalues have negative real part
    Eigen::Matrix4d A_cl = A_rp - B_rp * K;
    Eigen::EigenSolver<Eigen::Matrix4d> es(A_cl);
    bool stable = true;
    for (int i = 0; i < 4; ++i) {
        if (es.eigenvalues()(i).real() >= 0) {
            stable = false;
            break;
        }
    }
    return stable;
}

Eigen::Vector3d RollPitchLQR::compute(const Eigen::Vector4d& delta_x) const {
    return -K * delta_x;
}

void RollPitchLQR::printDiagnostics() const {
    std::cout << "\n=== Roll/Pitch LQR Diagnostics ===\n";
    std::cout << "Gain matrix K_rp (3x4):\n" << K << "\n\n";

    Eigen::Matrix4d A_cl = A_rp - B_rp * K;
    Eigen::EigenSolver<Eigen::Matrix4d> es(A_cl);
    std::cout << "Closed-loop eigenvalues of (A_rp - B_rp*K):\n";
    for (int i = 0; i < 4; ++i) {
        auto ev = es.eigenvalues()(i);
        std::cout << "  lambda_" << i << " = " << ev.real();
        if (std::abs(ev.imag()) > 1e-10)
            std::cout << " + " << ev.imag() << "j";
        std::cout << "  (stable: " << (ev.real() < 0 ? "YES" : "NO") << ")\n";
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> pev(P);
    std::cout << "\nP eigenvalues (should all be positive): "
              << pev.eigenvalues().transpose() << "\n";
    std::cout << "==================================\n\n";
}

bool RollPitchLQR::solveCARE() {
    // Solve CARE: A^T P + P A - P B R^{-1} B^T P + Q = 0
    //
    // Method: Matrix Sign Function applied to the Hamiltonian.
    //
    //   H = [ A,    -S   ]    where S = B R^{-1} B^T
    //       [-Q,    -A^T ]

    const int n = 4;
    const int n2 = 2 * n;  // 8
    Eigen::Matrix3d R_inv = R.inverse();
    Eigen::Matrix4d S = B_rp * R_inv * B_rp.transpose();

    // Build Hamiltonian (8x8)
    Eigen::MatrixXd Z(n2, n2);
    Z.block<4,4>(0, 0) =  A_rp;
    Z.block<4,4>(0, 4) = -S;
    Z.block<4,4>(4, 0) = -Q;
    Z.block<4,4>(4, 4) = -A_rp.transpose();

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
    Eigen::MatrixXd W11 = Z.block<4,4>(0, 0);
    Eigen::MatrixXd W21 = Z.block<4,4>(4, 0);
    Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(n, n);

    Eigen::MatrixXd P_raw = -W21 * (I_n - W11).inverse();

    // Symmetrise
    P = 0.5 * (P_raw + P_raw.transpose());

    // Verify P is positive definite
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> pev(P);
    double min_eig = pev.eigenvalues().minCoeff();
    if (min_eig < -1e-6) {
        std::cerr << "[CARE] P is not positive definite (min eigenvalue = "
                  << min_eig << ")\n";
        return false;
    }

    // Verify CARE residual: A^T P + P A - P S P + Q ≈ 0
    Eigen::Matrix4d residual =
        A_rp.transpose() * P + P * A_rp - P * S * P + Q;
    double res_norm = residual.norm();
    std::cout << "[CARE] Residual norm: " << res_norm << "\n";

    return res_norm < 1e-4;
}

// ============================================================================
// YawDamper
// ============================================================================

void YawDamper::build(const Config& cfg, const ControlAllocation& alloc) {
    k_r = cfg.yaw_damper.k_r;

    // The yaw row of B_tau is row 2 (z-axis torque).
    // We want to find delta_u such that B_tau_yaw * delta_u = tau_yaw_cmd
    // while minimising ||delta_u|| and minimising disturbance to roll/pitch.
    //
    // Full pseudoinverse of B_tau gives us the minimum-norm solution that
    // also distributes yaw torque with minimal roll/pitch coupling.
    // We extract the yaw (row 2) contribution.

    Eigen::Matrix3d B_tau_3x3 = alloc.B_tau;  // (3x3)

    // Pseudoinverse of B_tau: B_tau^+ = B_tau^T (B_tau B_tau^T)^{-1}
    Eigen::Matrix3d BtauBtauT = B_tau_3x3 * B_tau_3x3.transpose();
    Eigen::Matrix3d BtauBtauT_inv = BtauBtauT.inverse();
    Eigen::Matrix3d B_tau_pinv = B_tau_3x3.transpose() * BtauBtauT_inv;

    // Column 2 of the pseudoinverse maps yaw torque to motor commands
    yaw_pinv = B_tau_pinv.col(2);

    std::cout << "\n=== Yaw Damper Setup ===\n";
    std::cout << "k_r: " << k_r << "\n";
    std::cout << "Yaw row of B_tau: " << B_tau_3x3.row(2) << "\n";
    std::cout << "Pseudoinverse yaw column: " << yaw_pinv.transpose() << "\n";
    std::cout << "========================\n\n";
}

Eigen::Vector3d YawDamper::compute(double r) const {
    // Desired yaw torque: tau_yaw_cmd = -k_r * r
    double tau_yaw_cmd = -k_r * r;
    // Convert to motor commands via pseudoinverse
    return yaw_pinv * tau_yaw_cmd;
}

void YawDamper::printDiagnostics() const {
    std::cout << "\n=== Yaw Damper Diagnostics ===\n";
    std::cout << "k_r = " << k_r << "\n";
    std::cout << "yaw_pinv = " << yaw_pinv.transpose() << "\n";
    std::cout << "==============================\n\n";
}

// ============================================================================
// HeadingAwareCommand
// ============================================================================

Eigen::Vector2d HeadingAwareCommand::computeAttitudeCmd(
    double ax_desired, double ay_desired, double psi)
{
    const double g = 9.81;
    double cpsi = std::cos(psi);
    double spsi = std::sin(psi);

    double phi_cmd   = ( cpsi * ax_desired + spsi * ay_desired) / g;
    double theta_cmd = (-spsi * ax_desired + cpsi * ay_desired) / g;

    return Eigen::Vector2d(phi_cmd, theta_cmd);
}
