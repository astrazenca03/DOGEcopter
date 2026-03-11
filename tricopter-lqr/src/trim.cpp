#include "trim.hpp"
#include <iostream>
#include <cmath>

bool TrimSolver::solve(const Config& cfg, const ControlAllocation& alloc) {
    const int N = static_cast<int>(cfg.rotors.size());
    const double g = 9.81;
    const double mg = cfg.mass * g;

    // Find u_hover minimising ||B_tau * u||^2 subject to c^T u = mg.
    //
    // For a servoless tricopter with asymmetric spin dirs [+1,-1,+1],
    // exact zero torque is physically impossible (3 actuators, 4 constraints).
    // The KKT solution finds the minimum-norm-torque hover point.
    //
    // KKT: [2M c; c^T 0] [u; lambda] = [0; mg]  where M = B_tau^T B_tau

    Eigen::VectorXd c(N);
    for (int i = 0; i < N; ++i)
        c(i) = cfg.rotors[i].k_T * cfg.rotors[i].thrust_axis.z();

    Eigen::MatrixXd M = alloc.B_tau.transpose() * alloc.B_tau;

    Eigen::MatrixXd KKT(N + 1, N + 1);
    Eigen::VectorXd rhs(N + 1);
    KKT.topLeftCorner(N, N) = 2.0 * M;
    KKT.topRightCorner(N, 1) = c;
    KKT.bottomLeftCorner(1, N) = c.transpose();
    KKT(N, N) = 0.0;
    rhs.head(N).setZero();
    rhs(N) = mg;

    Eigen::VectorXd sol = KKT.colPivHouseholderQr().solve(rhs);
    u_hover = sol.head(N);
    total_thrust_hover = c.dot(u_hover);

    Eigen::Vector3d residual_torque = alloc.B_tau * u_hover;
    double thrust_err = std::abs(total_thrust_hover - mg);

    std::cout << "\n=== Trim Solver Results ===\n";
    std::cout << "u_hover (omega_i^2):\n";
    for (int i = 0; i < N; ++i) {
        double omega = std::sqrt(std::max(0.0, u_hover(i)));
        std::cout << "  Rotor " << i << " (" << cfg.rotors[i].name << "): "
                  << "omega^2 = " << u_hover(i)
                  << "  omega = " << omega << " rad/s\n";
    }
    std::cout << "Total hover thrust: " << total_thrust_hover
              << " N  (target: " << mg << " N)\n";
    std::cout << "Thrust error: " << thrust_err << " N\n";
    std::cout << "Residual torque: " << residual_torque.transpose() << " Nm\n";
    std::cout << "Residual torque norm: " << residual_torque.norm() << " Nm\n";

    double omega_max_sq = cfg.omega_max * cfg.omega_max;
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (u_hover(i) < 0) {
            std::cout << "*** WARNING: Rotor " << i << " has negative omega^2! ***\n";
            valid = false;
        }
        if (u_hover(i) > omega_max_sq) {
            std::cout << "*** WARNING: Rotor " << i << " exceeds omega_max^2! ***\n";
            valid = false;
        }
    }

    if (residual_torque.norm() > 0.001) {
        std::cout << "NOTE: Nonzero residual torque (primarily yaw) is expected for\n"
                  << "a servoless tricopter with " << N << " fixed-spin rotors. The\n"
                  << "controller treats this as a constant disturbance.\n";
    }
    std::cout << "===========================\n\n";

    return valid && thrust_err < 0.01;
}
