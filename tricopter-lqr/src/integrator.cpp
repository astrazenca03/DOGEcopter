#include "integrator.hpp"
#include "dynamics.hpp"

namespace integrator {

Eigen::VectorXd rk4Step(
    const DerivFunc& f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    const Config& cfg,
    double dt,
    const Eigen::Vector3d& external_torque)
{
    // Classical 4th-order Runge-Kutta
    Eigen::VectorXd k1 = f(x, u, cfg, external_torque);
    Eigen::VectorXd k2 = f(x + 0.5 * dt * k1, u, cfg, external_torque);
    Eigen::VectorXd k3 = f(x + 0.5 * dt * k2, u, cfg, external_torque);
    Eigen::VectorXd k4 = f(x + dt * k3, u, cfg, external_torque);

    Eigen::VectorXd x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    // Renormalise quaternion after RK4 update
    dynamics::normaliseQuaternion(x_new);

    return x_new;
}

}  // namespace integrator
