#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include <Eigen/Dense>

/// Roll/pitch LQR controller (4-state inner loop).
///
/// 4-state system (yaw removed):
///   delta_x = [delta_phi, delta_theta, delta_p, delta_q]
/// 3 inputs:
///   delta_u = [delta_omega1^2, delta_omega2^2, delta_omega3^2]
///
/// A_rp (4x4):  [0_{2x2},  I_{2x2}]    (at hover, omega_0 = 0)
///              [0_{2x2},  0_{2x2}]
///
/// B_rp (4x3):  [0_{2x3}                          ]
///              [first 2 rows of J_inv * B_tau     ]
///
/// Solves CARE for this reduced system.
struct RollPitchLQR {
    Eigen::Matrix4d A_rp;                   // linearised A (4x4)
    Eigen::Matrix<double, 4, 3> B_rp;      // linearised B (4x3)
    Eigen::Matrix4d Q;                      // state cost (4x4 diagonal)
    Eigen::Matrix3d R;                      // input cost (3x3 diagonal)
    Eigen::Matrix4d P;                      // CARE solution (4x4)
    Eigen::Matrix<double, 3, 4> K;         // gain matrix (3x4)

    /// Build A, B, Q, R from config and allocation, then solve CARE.
    /// Returns true if all eigenvalues have negative real part.
    bool build(const Config& cfg, const ControlAllocation& alloc);

    /// Compute control: delta_u = -K * [delta_phi, delta_theta, delta_p, delta_q]
    Eigen::Vector3d compute(const Eigen::Vector4d& delta_x) const;

    /// Print diagnostics: K, eigenvalues.
    void printDiagnostics() const;

private:
    /// Solve CARE via matrix sign function on the Hamiltonian.
    bool solveCARE();
};

/// Yaw rate damper: simple proportional control on yaw rate.
///
/// Computes desired yaw torque: tau_yaw_cmd = -k_r * r
/// Converts to motor commands using pseudoinverse of the yaw row of B_tau,
/// minimising roll/pitch disturbance.
struct YawDamper {
    double k_r;                         // proportional gain
    Eigen::Vector3d yaw_pinv;           // pseudoinverse of yaw row of B_tau (3x1)

    /// Build from config and allocation.
    void build(const Config& cfg, const ControlAllocation& alloc);

    /// Compute motor command increment for yaw damping.
    /// r = current yaw rate (rad/s)
    Eigen::Vector3d compute(double r) const;

    void printDiagnostics() const;
};

/// Heading-aware position command infrastructure.
///
/// Converts desired translational acceleration (world frame) to
/// commanded roll/pitch angles, accounting for current heading.
///
/// phi_cmd   =  (cos(psi)*ax + sin(psi)*ay) / g
/// theta_cmd = (-sin(psi)*ax + cos(psi)*ay) / g
///
/// For now, desired acceleration is zero (level hover), but the
/// infrastructure exists for a future outer position loop.
struct HeadingAwareCommand {
    /// Compute commanded [phi, theta] from desired world-frame acceleration
    /// and current yaw angle.
    static Eigen::Vector2d computeAttitudeCmd(
        double ax_desired, double ay_desired, double psi);
};
