#include "simulation.hpp"
#include "dynamics.hpp"
#include "integrator.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

void Simulation::run(
    const Config& cfg,
    const ControlAllocation& /* alloc */,
    const TrimSolver& trim,
    const AttitudeLQR& lqr,
    const std::string& output_path)
{
    const int N = static_cast<int>(cfg.rotors.size());
    const double dt = cfg.sim.dt;
    const int num_steps = static_cast<int>(cfg.sim.duration / dt);
    const double omega_max_sq = cfg.omega_max * cfg.omega_max;
    const double mg = cfg.mass * 9.81;

    double k_T_avg = 0;
    for (int i = 0; i < N; ++i)
        k_T_avg += cfg.rotors[i].k_T;
    k_T_avg /= N;

    // Altitude PID
    PIDController alt_pid;
    alt_pid.init(cfg.alt_pid);

    // Initial state
    Eigen::VectorXd x = dynamics::buildInitialState(cfg);

    // Derivative function for integrator
    integrator::DerivFunc deriv_func = dynamics::computeDerivative;

    // Open CSV
    std::ofstream csv(output_path);
    csv << "time,x,y,z,vx,vy,vz,phi_deg,theta_deg,psi_deg,p,q,r,u1,u2,u3,T_total\n";

    std::cout << "\n=== Running Simulation ===\n";
    std::cout << "Duration: " << cfg.sim.duration << " s, dt: " << dt << " s\n";
    std::cout << "Initial perturbation: roll=" << cfg.sim.initial_roll_deg
              << " deg, pitch=" << cfg.sim.initial_pitch_deg
              << " deg, yaw=" << cfg.sim.initial_yaw_deg << " deg\n\n";

    for (int step = 0; step <= num_steps; ++step) {
        double t = step * dt;

        // --- Extract current attitude ---
        Eigen::Vector3d euler = dynamics::quaternionToEulerZYX(x);
        Eigen::Vector3d omega_body = x.segment<3>(10);

        // --- Attitude error ---
        // Wrap yaw error to [-pi, pi] for large-angle robustness
        double psi_err = euler(2);
        while (psi_err > M_PI)  psi_err -= 2.0 * M_PI;
        while (psi_err < -M_PI) psi_err += 2.0 * M_PI;

        Eigen::Matrix<double, 6, 1> delta_x_att;
        delta_x_att(0) = euler(0);   // phi error
        delta_x_att(1) = euler(1);   // theta error
        delta_x_att(2) = psi_err;    // psi error (wrapped)
        delta_x_att.tail<3>() = omega_body;

        // --- Inner loop: attitude LQR ---
        Eigen::Vector3d delta_u_att = lqr.compute(delta_x_att);

        // --- Outer loop: altitude PID ---
        double z_current = x(2);
        double z_error = cfg.sim.z_desired - z_current;
        double pid_out = alt_pid.update(z_error, z_current, dt);
        double T_cmd = mg + pid_out;

        // Collective: distribute thrust command equally
        double delta_u_coll = (T_cmd - trim.total_thrust_hover) / (N * k_T_avg);

        // --- Combined control law ---
        // u = u_hover + delta_u_att + delta_u_coll * [1,1,...,1]
        Eigen::VectorXd u(N);
        for (int i = 0; i < N; ++i) {
            u(i) = trim.u_hover(i) + delta_u_att(i) + delta_u_coll;
            u(i) = std::clamp(u(i), 0.0, omega_max_sq);
        }

        // --- Total thrust for logging ---
        double T_total = 0;
        for (int i = 0; i < N; ++i)
            T_total += cfg.rotors[i].k_T * u(i);

        // --- Log to CSV ---
        double phi_deg   = euler(0) * 180.0 / M_PI;
        double theta_deg = euler(1) * 180.0 / M_PI;
        double psi_deg   = euler(2) * 180.0 / M_PI;

        csv << std::fixed << std::setprecision(6)
            << t << ","
            << x(0) << "," << x(1) << "," << x(2) << ","
            << x(3) << "," << x(4) << "," << x(5) << ","
            << phi_deg << "," << theta_deg << "," << psi_deg << ","
            << omega_body(0) << "," << omega_body(1) << "," << omega_body(2) << ",";
        for (int i = 0; i < N; ++i)
            csv << std::sqrt(std::max(0.0, u(i))) << ",";
        csv << T_total << "\n";

        // Print samples
        if (step == 0 || step == 1 || step == 2 ||
            step == num_steps - 2 || step == num_steps - 1 || step == num_steps ||
            (step % 1000 == 0)) {
            std::cout << "t=" << std::fixed << std::setprecision(3) << t
                      << "  phi=" << std::setw(8) << std::setprecision(3) << phi_deg
                      << "  theta=" << std::setw(8) << theta_deg
                      << "  psi=" << std::setw(8) << psi_deg
                      << "  z=" << std::setw(8) << std::setprecision(4) << x(2)
                      << "  T=" << std::setw(8) << std::setprecision(3) << T_total
                      << "\n";
        }

        // --- Disturbance ---
        Eigen::Vector3d ext_torque = Eigen::Vector3d::Zero();
        if (t >= cfg.sim.disturbance_time &&
            t < cfg.sim.disturbance_time + cfg.sim.disturbance_duration) {
            ext_torque = cfg.sim.disturbance_torque;
        }

        // --- Integrate ---
        if (step < num_steps) {
            x = integrator::rk4Step(deriv_func, x, u, cfg, dt, ext_torque);
        }
    }

    csv.close();
    std::cout << "\nSimulation complete. Output: " << output_path << "\n";
    std::cout << "===========================\n";
}
