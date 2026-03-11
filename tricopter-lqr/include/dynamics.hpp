#pragma once

#include "config.hpp"
#include <Eigen/Dense>

/// 13-state rigid body dynamics for a multi-rotor.
///
/// State vector x (13 x 1):
///   [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
///    0-2: position (world frame)
///    3-5: velocity (world frame)
///    6-9: quaternion (w,x,y,z) body-to-world
///   10-12: angular velocity (body frame)
///
/// Control input u (N x 1): omega_i^2 for each rotor.
///
/// Gravity convention: g = [0, 0, -9.81] in world frame (z-up).

namespace dynamics {

/// Compute state derivative dx/dt given state x, control u, and config.
/// Optional external_torque adds a body-frame torque disturbance.
Eigen::VectorXd computeDerivative(
    const Eigen::VectorXd& x,         // (13) state
    const Eigen::VectorXd& u,         // (N) control: omega_i^2
    const Config& cfg,
    const Eigen::Vector3d& external_torque = Eigen::Vector3d::Zero());

/// Normalise the quaternion component of state vector in-place.
void normaliseQuaternion(Eigen::VectorXd& x);

/// Extract Euler angles (ZYX convention) from quaternion in state.
/// Returns [roll, pitch, yaw] in radians.
Eigen::Vector3d quaternionToEulerZYX(const Eigen::VectorXd& x);

/// Build initial state from config (position at origin, perturbed attitude).
Eigen::VectorXd buildInitialState(const Config& cfg);

}  // namespace dynamics
