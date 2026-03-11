#include "dynamics.hpp"
#include <cmath>

namespace dynamics {

Eigen::VectorXd computeDerivative(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    const Config& cfg,
    const Eigen::Vector3d& external_torque)
{
    // State extraction
    // x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    Eigen::Vector3d pos = x.segment<3>(0);
    Eigen::Vector3d vel = x.segment<3>(3);
    Eigen::Quaterniond quat(x(6), x(7), x(8), x(9));  // (w, x, y, z)
    quat.normalize();
    Eigen::Vector3d omega = x.segment<3>(10);           // body-frame angular velocity

    (void)pos;  // position not needed for derivative computation

    const int N = static_cast<int>(cfg.rotors.size());

    // Compute total force and torque in body frame
    Eigen::Vector3d F_body = Eigen::Vector3d::Zero();
    Eigen::Vector3d tau_body = Eigen::Vector3d::Zero();

    for (int i = 0; i < N; ++i) {
        const auto& r = cfg.rotors[i];
        double omega_sq = u(i);

        // Thrust force: T_i = k_T * omega_i^2, along thrust_axis
        Eigen::Vector3d T_i = r.k_T * omega_sq * r.thrust_axis;

        // Drag torque: Q_i = k_Q * omega_i^2, along spin_axis, direction = spin_dir
        Eigen::Vector3d Q_i = static_cast<double>(r.spin_direction) * r.k_Q * omega_sq * r.spin_axis;

        F_body += T_i;
        tau_body += r.position.cross(T_i) + Q_i;
    }
    tau_body += external_torque;

    // Rotation matrix: body -> world
    Eigen::Matrix3d R_bw = quat.toRotationMatrix();

    // Translational dynamics (world frame):
    // m * a = R * F_body + m * g
    // Gravity: [0, 0, -9.81] in world frame (z-up)
    Eigen::Vector3d gravity(0.0, 0.0, -9.81);
    Eigen::Vector3d accel = (R_bw * F_body) / cfg.mass + gravity;

    // Rotational dynamics (body frame, Euler's equation):
    // J * omega_dot = tau - omega x (J * omega)
    Eigen::Vector3d J_omega = cfg.J * omega;
    Eigen::Vector3d omega_dot = cfg.J.inverse() * (tau_body - omega.cross(J_omega));

    // Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
    Eigen::Quaterniond omega_quat(0.0, omega.x(), omega.y(), omega.z());
    Eigen::Quaterniond q_dot;
    // q_dot = 0.5 * q * omega_quat (Hamilton product)
    q_dot.w() = 0.5 * (-quat.x()*omega.x() - quat.y()*omega.y() - quat.z()*omega.z());
    q_dot.x() = 0.5 * ( quat.w()*omega.x() + quat.y()*omega.z() - quat.z()*omega.y());
    q_dot.y() = 0.5 * ( quat.w()*omega.y() - quat.x()*omega.z() + quat.z()*omega.x());
    q_dot.z() = 0.5 * ( quat.w()*omega.z() + quat.x()*omega.y() - quat.y()*omega.x());

    // Assemble derivative
    Eigen::VectorXd dx(13);
    dx.segment<3>(0) = vel;            // d(pos)/dt = vel
    dx.segment<3>(3) = accel;          // d(vel)/dt = accel
    dx(6) = q_dot.w();
    dx(7) = q_dot.x();
    dx(8) = q_dot.y();
    dx(9) = q_dot.z();
    dx.segment<3>(10) = omega_dot;     // d(omega)/dt

    return dx;
}

void normaliseQuaternion(Eigen::VectorXd& x) {
    double norm = std::sqrt(x(6)*x(6) + x(7)*x(7) + x(8)*x(8) + x(9)*x(9));
    if (norm > 1e-10) {
        x(6) /= norm;
        x(7) /= norm;
        x(8) /= norm;
        x(9) /= norm;
    }
}

Eigen::Vector3d quaternionToEulerZYX(const Eigen::VectorXd& x) {
    // ZYX convention: yaw(psi) -> pitch(theta) -> roll(phi)
    double qw = x(6), qx = x(7), qy = x(8), qz = x(9);

    // Roll (phi) - rotation about x
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    double phi = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (theta) - rotation about y
    double sinp = 2.0 * (qw * qy - qz * qx);
    double theta;
    if (std::abs(sinp) >= 1.0)
        theta = std::copysign(M_PI / 2.0, sinp);
    else
        theta = std::asin(sinp);

    // Yaw (psi) - rotation about z
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double psi = std::atan2(siny_cosp, cosy_cosp);

    return Eigen::Vector3d(phi, theta, psi);
}

Eigen::VectorXd buildInitialState(const Config& cfg) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(13);

    // Position at origin
    // Velocity zero

    // Initial attitude from Euler angles (ZYX: yaw, pitch, roll)
    double phi   = cfg.sim.initial_roll_deg  * M_PI / 180.0;
    double theta = cfg.sim.initial_pitch_deg * M_PI / 180.0;
    double psi   = cfg.sim.initial_yaw_deg   * M_PI / 180.0;

    // Build quaternion from ZYX Euler angles
    Eigen::Quaterniond q =
        Eigen::AngleAxisd(psi,   Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(phi,   Eigen::Vector3d::UnitX());
    q.normalize();

    x(6) = q.w();
    x(7) = q.x();
    x(8) = q.y();
    x(9) = q.z();

    // Angular rates zero
    return x;
}

}  // namespace dynamics
