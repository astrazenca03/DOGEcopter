#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include <Eigen/Dense>

/// Hover trim solver.
///
/// Finds u_hover = [omega_1^2, ..., omega_N^2] such that:
///   - Total thrust = m*g  (along gravity axis)
///   - Net torque = 0      (B_tau * u_hover = 0)
///
/// For asymmetric spin directions this yields UNEQUAL motor speeds.
struct TrimSolver {
    Eigen::VectorXd u_hover;     // (N) trim omega_i^2 values
    double total_thrust_hover;   // should equal m*g

    /// Solve trim. Returns true on success. Prints results.
    bool solve(const Config& cfg, const ControlAllocation& alloc);
};
