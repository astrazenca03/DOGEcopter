#pragma once

#include "config.hpp"
#include <Eigen/Dense>
#include <functional>

namespace integrator {

/// State derivative function signature.
/// Takes (state, control, config, external_torque) -> dx/dt.
using DerivFunc = std::function<Eigen::VectorXd(
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    const Config&,
    const Eigen::Vector3d&)>;

/// Classical 4th-order Runge-Kutta integrator.
///
/// Advances state x by one timestep dt using control input u.
/// Quaternion is renormalised after the final RK4 update.
Eigen::VectorXd rk4Step(
    const DerivFunc& f,
    const Eigen::VectorXd& x,         // (13) current state
    const Eigen::VectorXd& u,         // (N) control input
    const Config& cfg,
    double dt,
    const Eigen::Vector3d& external_torque = Eigen::Vector3d::Zero());

}  // namespace integrator
