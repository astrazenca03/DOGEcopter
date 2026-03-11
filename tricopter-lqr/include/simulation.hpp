#pragma once

#include "config.hpp"
#include "control_allocation.hpp"
#include "trim.hpp"
#include "lqr.hpp"
#include "pid.hpp"
#include <Eigen/Dense>
#include <string>

/// Closed-loop simulation: roll/pitch LQR + yaw damper + altitude PID.
///
/// Each timestep:
///   1. Extract Euler angles & angular rates from state
///   2. Heading-aware command -> desired [phi, theta] (zero for hover)
///   3. Roll/pitch error -> LQR -> delta_u_rollpitch
///   4. Yaw rate -> damper -> delta_u_yaw
///   5. Altitude error -> PID -> T_cmd -> delta_u_coll
///   6. u = u_hover + delta_u_rollpitch + delta_u_yaw + delta_u_coll * [1,1,...,1]
///   7. Clamp to [0, omega_max^2]
///   8. RK4 integrate
///   9. Log to CSV
struct Simulation {
    /// Run full simulation. Writes CSV to output_path.
    void run(const Config& cfg,
             const ControlAllocation& alloc,
             const TrimSolver& trim,
             const RollPitchLQR& lqr,
             const YawDamper& yaw_damper,
             const std::string& output_path = "sim_output.csv");
};
