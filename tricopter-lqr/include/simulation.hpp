#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include "trim.hpp"
#include "lqr.hpp"
#include "pid.hpp"
#include <Eigen/Dense>
#include <string>

/// Closed-loop simulation: attitude LQR (inner) + altitude PID (outer).
///
/// Each timestep:
///   1. Extract Euler angles & angular rates from state
///   2. Attitude error -> LQR -> delta_u_att
///   3. Altitude error -> PID -> T_cmd -> delta_u_coll
///   4. u = u_hover + delta_u_att + delta_u_coll * [1,1,...,1]
///   5. Clamp to [0, omega_max^2]
///   6. RK4 integrate
///   7. Log to CSV
struct Simulation {
    /// Run full simulation. Writes CSV to output_path.
    void run(const Config& cfg,
             const ControlAllocation& alloc,
             const TrimSolver& trim,
             const AttitudeLQR& lqr,
             const std::string& output_path = "sim_output.csv");
};
