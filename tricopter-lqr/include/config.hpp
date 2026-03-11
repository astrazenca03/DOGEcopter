#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

/// Per-rotor configuration
struct RotorConfig {
    std::string name;
    Eigen::Vector3d position;      // position from CG [m], body frame
    Eigen::Vector3d thrust_axis;   // unit vector, thrust direction
    Eigen::Vector3d spin_axis;     // unit vector, spin axis
    int spin_direction;            // +1 CW, -1 CCW
    double k_T;                    // thrust coeff: T = k_T * omega^2
    double k_Q;                    // drag torque coeff: Q = k_Q * omega^2
};

/// Roll/pitch LQR tuning (4-state: phi, theta, p, q)
struct AttitudeLQRConfig {
    Eigen::Vector4d Q_diag;  // 4 diagonal weights: [q_phi, q_theta, q_p, q_q]
    Eigen::Vector3d R_diag;  // 3 diagonal weights for R
};

/// Yaw rate damper tuning
struct YawDamperConfig {
    double k_r;  // proportional gain on yaw rate
};

/// Altitude PID tuning
struct AltitudePIDConfig {
    double Kp;
    double Ki;
    double Kd;
    double integral_limit;
    double output_limit;
};

/// Simulation parameters
struct SimConfig {
    double dt;
    double duration;
    double initial_roll_deg;
    double initial_pitch_deg;
    double initial_yaw_deg;
    double z_desired;
    double disturbance_time;
    double disturbance_duration;
    Eigen::Vector3d disturbance_torque;  // Nm, body frame
};

/// Top-level configuration
struct Config {
    double mass;                      // kg
    Eigen::Matrix3d J;                // full 3x3 inertia tensor
    std::vector<RotorConfig> rotors;  // variable number of rotors
    AttitudeLQRConfig att_lqr;
    YawDamperConfig yaw_damper;
    AltitudePIDConfig alt_pid;
    SimConfig sim;
    double omega_max;                 // motor speed limit [rad/s]
};

/// Load configuration from a YAML file. Throws on parse error.
Config loadConfig(const std::string& filepath);
