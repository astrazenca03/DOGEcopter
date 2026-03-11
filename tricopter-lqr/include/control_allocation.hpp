#pragma once

#include "config.hpp"
#include <Eigen/Dense>

/// Control allocation: maps motor omega^2 to body forces & torques.
///
/// B_alloc is 6 x N where N = number of rotors.
///   Rows 0-2: force in body frame    F = B_force * u
///   Rows 3-5: torque in body frame   tau = B_tau * u
/// where u = [omega_1^2, omega_2^2, ..., omega_N^2]
struct ControlAllocation {
    Eigen::MatrixXd B_alloc;  // (6 x N) full allocation matrix
    Eigen::MatrixXd B_force;  // (3 x N) force sub-block
    Eigen::MatrixXd B_tau;    // (3 x N) torque sub-block

    /// Build from config. Prints rank, condition number, eigenvalues of B_tau.
    void build(const Config& cfg);

    /// Print diagnostics: B_tau, rank, condition number, eigenvalues.
    void printDiagnostics() const;
};
